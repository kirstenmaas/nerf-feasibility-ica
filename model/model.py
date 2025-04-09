import torch
import torch.nn as nn
import numpy as np
import sys
import pdb

sys.path.append('model/')

from model.barf_helpers import get_cam_matrices_table, randomize_depth, get_minibatches

class Model(nn.Module):
    """
    A CPPN model, mapping a number of input coordinates to a multidimensional output (e.g. color)
    """
    def __init__(self, model_definition: dict) -> None:
        """
        Args:
            model_definition: dictionary containing all the needed parameters
                - num_layers: number of hidden layers
                - num_filters: number of filters in the hidden blocks
                - num_input_channels: number of expected input channels
                - num_input_channels_views: number of expected input channels for the view direction
                - num_output_channels: number of expected output channels
                - use_bias: whether biases are used
                - pos_enc: which positional encoding to apply: 'none', 'fourier', 'barf'
                - pos_enc_basis: basis for positional encoding (L)
                - pos_enc_basis_views: basis for positional encoding for views (L)
        """
        super().__init__()
        self.version = "v0.00"
        self.model_definition = model_definition
        self.device = model_definition['device']

        # getting the parameters
        self.num_early_layers = model_definition['num_early_layers']
        self.num_late_layers = model_definition['num_late_layers']
        self.num_filters = model_definition['num_filters']
        self.num_input_channels = model_definition['num_input_channels'] # x,y,z
        self.num_input_channels_views = model_definition['num_input_channels_views'] # direction unit vector
        self.num_output_channels = model_definition['num_output_channels']
        self.use_bias = model_definition['use_bias']
        self.use_pos_enc = model_definition['pos_enc']
        self.act_func = model_definition['act_func']

        # camera related
        self.pose_optimization = model_definition['pose_optimization'] #whether to apply pose optimization or not
        self.num_train_cases = model_definition['num_train_cases']
        self.poses = model_definition['poses'] # dict of poses based on proj_ids
        self.source_pts = model_definition['source_pts'] #dict of source pts based on proj ids
        self.scale_factor = model_definition['scale_factor']
        self.learn_iso_center = model_definition['learn_iso_center']
        self.learn_table = model_definition['learn_table']
        self.learn_rotation = model_definition['learn_rotation']
        self.learn_intensity = model_definition['learn_intensity']
        self.x_ray_type = model_definition['x_ray_type']

        # pose shifts
        # [0,1,2] theta, phi, larm in rad
        # [3, 4, 5] translation iso
        # [6, 7, 8] translation table
        self.pose_shifts = nn.ParameterDict()
        init_shift = torch.Tensor([0.,0.,0.,0.,0.,0.,0.,0.,0.])
        for i, key in enumerate(self.poses):
            self.pose_shifts[key] = init_shift
            if i == 0:
                self.pose_shifts[key].requires_grad = False

        self.initial_intensities = nn.ParameterDict() if self.learn_intensity else {}
        for key in self.poses:
            init_intensity = torch.Tensor([1.]).to(self.device)
            self.initial_intensities[key] = nn.Parameter(init_intensity) if self.learn_intensity else init_intensity

        # ray sampling
        self.direction = model_definition['direction'] #camera direction
        self.z_vals = model_definition['z_vals'] #camera depth vals
        self.depth_samples_per_ray = model_definition['depth_samples_per_ray']
        self.near_thresh = model_definition['near_thresh']
        self.far_thresh = model_definition['far_thresh']
        self.chunksize = model_definition['chunksize']

        self.enc_fun = None

        self.first_act_func = nn.ReLU()
        self.act_func = nn.ReLU()

        self.input_features_pts = self.num_input_channels

        if self.use_pos_enc != 'none':
            self.pos_enc_basis = model_definition['pos_enc_basis']
            self.input_features_pts = self.num_input_channels + self.num_input_channels * 2 * self.pos_enc_basis

            if self.use_pos_enc == 'fourier':
                self.input_features_pts = self.num_input_channels * 2 * self.pos_enc_basis
                self.fourier_sigma = model_definition['fourier_sigma']
                self.fourier_coefficients = (model_definition['fourier_gaussian'] * self.fourier_sigma).to(self.device)

            self.windowed_alpha = 0

        self.input_features = self.input_features_pts
        
        # model understanding API
        self.store_activations = False
        self.activation_dictionary = {}

        self.create_net()

    def create_net(self):
        input_features = self.input_features
        num_filters = self.num_filters
        use_bias = self.use_bias
        num_output_channels = self.num_output_channels

        # creating the learnable blocks
        early_pts_layers = []
        # input layer
        early_pts_layers += self.__create_layer(self.input_features, num_filters,
                                           use_bias, activation=self.first_act_func)
        # hidden layers: early
        for _ in range(self.num_early_layers):
            early_pts_layers += self.__create_layer(num_filters, num_filters,
                                               use_bias, activation=self.act_func)

        self.early_pts_layers = nn.ModuleList(early_pts_layers)

        # skip connection
        if self.num_late_layers > 0:
            self.skip_connection = self.__create_layer(num_filters + input_features, num_filters,
                                                use_bias, activation=self.act_func)

            late_pts_layers = []
            for _ in range(self.num_late_layers - 1):
                late_pts_layers += self.__create_layer(num_filters, num_filters,
                                                use_bias, activation=self.act_func)

            self.late_pts_layers = nn.ModuleList(late_pts_layers)
        
        # output layer
        self.output_linear = self.__create_layer(num_filters, num_output_channels, use_bias, activation=None)

    @staticmethod
    def __create_layer(num_in_filters: int, num_out_filters: int,
                       use_bias: bool, activation=nn.ReLU()) -> nn.Sequential:
        block = []
        block.append(nn.Linear(num_in_filters, num_out_filters, bias=use_bias)) # Dense layer
        if activation:
            block.append(activation)
        block = nn.Sequential(*block)

        return block

    def activations(self, store_activations: bool) -> None:
        """
        Configure the model to retain or discard the activations during the forward pass

        Args:
            activations (bool): keep/discard the activations during inference
        """

        self.store_activations = store_activations

        if not store_activations:
            self.activation_dictionary = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input_pts, _ = torch.split(x, [self.num_input_channels, self.num_input_channels_views], dim=-1)
        
        values = input_pts

        pos_enc = self.use_pos_enc
        pts_encoded = input_pts
        if pos_enc != 'none':
            pts_encoded = self.pos_enc(input_pts, self.pos_enc_basis, 'pts')

        values = pts_encoded
        for _, pts_layer in enumerate(self.early_pts_layers):
            values = pts_layer(values)

        if self.num_late_layers > 0:
            values = self.skip_connection(torch.cat([pts_encoded, values], dim=-1))
            for _, pts_layer in enumerate(self.late_pts_layers):
                values = pts_layer(values)

        outputs = self.output_linear(values)

        return outputs
    
    def forward_poses_eval(self, proj_id) -> torch.Tensor:
        proj_pose = self.poses[proj_id]
        pose_shift = self.pose_shifts[proj_id]
        source_pt = self.source_pts[proj_id]

        src_matrix = get_cam_matrices_table(proj_pose[None, :] + pose_shift[None, :], source_pt[None, :], self.x_ray_type).to(self.device)[0]

        ray_directions = torch.sum(self.direction[..., None, :] * src_matrix[:3, :3], dim=-1).reshape((-1, 3)).float()
        ray_origins = src_matrix[:3, -1].expand(ray_directions.shape).to(self.device).reshape((-1, 3)).float()

        depth_values = randomize_depth(self.z_vals, self.device)
        query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
        qry_pts = query_points.reshape((-1, 3)).float()

        outputs = self.get_batched_outputs(qry_pts)
        return outputs, src_matrix, ray_directions, depth_values
    
    def get_batched_outputs(self, qry_pts):
        outputs = []
        batches = get_minibatches(qry_pts, chunksize=self.chunksize)
        for batch in batches:
            outputs.append(self.forward(batch))
        outputs = torch.cat(outputs, dim=0)
        return outputs
    
    def forward_poses(self, proj_id, sample_row=[], sample_column=[]) -> torch.Tensor:
        # add learned shift
        if self.pose_optimization:
            pose_shifts = self.pose_shifts[proj_id].clone()
            if not self.learn_iso_center:
                pose_shifts[:, 3:6] = 0
            if not self.learn_table:
                pose_shifts[:, 6:9] = 0
            if not self.learn_rotation:
                pose_shifts[:, :3] = 0

            pose = self.poses[proj_id] + pose_shifts
        else: 
            pose = self.poses[proj_id]

        source_pt = self.source_pts[proj_id]

        # 6 DOF to source matrix
        src_matrix = get_cam_matrices_table(pose[None, :], source_pt, self.x_ray_type).to(self.device).float()[0]

        # retrieve samples
        if len(sample_row) > 0 and len(sample_column) > 0:
            # sample a number of query points
            direction = self.direction[sample_row, sample_column].reshape(len(sample_row), self.direction.shape[-1])
        else:
            direction = self.direction
        direction = direction[..., None, :]

        ray_directions = torch.sum(direction * src_matrix[:3, :3], dim=-1).float()
        ray_origins = src_matrix[:3, -1].expand(ray_directions.shape).to(self.device).float()

        depth_values = randomize_depth(self.z_vals, self.device)
        query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
        qry_pts = query_points.reshape((-1, 3)).float()
       
        outputs = self.get_batched_outputs(qry_pts)
        return outputs, src_matrix, ray_directions, depth_values

    def forward_pixel_poses(self, proj_ids, sample_row=[], sample_column=[]) -> torch.Tensor:
        
        unique_proj_ids = np.unique(proj_ids)
        proj_ids_idx = np.array([np.argwhere(unique_proj_ids == proj_id) for proj_id in proj_ids]).flatten()

        if self.pose_optimization:
            pose_shifts_lst = []
            for proj_id in unique_proj_ids:
                pose_shifts_lst.append(self.pose_shifts[proj_id])

            pose_shifts = torch.stack(pose_shifts_lst)

           # only learn rotations
            if not self.learn_iso_center:
                pose_shifts[:, 3:6] = 0
            
            if not self.learn_table:
                pose_shifts[:, 6:9] = 0

            if not self.learn_rotation:
                pose_shifts[:, :3] = 0

            unique_poses = torch.stack([self.poses[proj_id] + pose_shifts[i] for i, proj_id in enumerate(unique_proj_ids)])
        else: 
            unique_poses = torch.stack([self.poses[proj_id] for proj_id in unique_proj_ids])
        
        source_pts = torch.stack([self.source_pts[proj_id] for proj_id in unique_proj_ids])

        # 6 DOF to source matrix
        unique_src_matrices = get_cam_matrices_table(unique_poses, source_pts, self.x_ray_type).to(self.device)
        src_matrices = unique_src_matrices[proj_ids_idx]

        # retrieve samples
        if len(sample_row) > 0 and len(sample_column) > 0:
            # sample a number of query points
            direction = self.direction[sample_row, sample_column]
        else:
            direction = self.direction
        direction = direction[..., None, :]

        ray_directions = torch.sum(direction * src_matrices[:, :3, :3], dim=-1).float()
        ray_origins = src_matrices[:, :3, -1].expand(ray_directions.shape).to(self.device).float()

        depth_values = randomize_depth(self.z_vals, self.device)
        query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
        qry_pts = query_points.reshape((-1, 3)).float()

        outputs = self.get_batched_outputs(qry_pts)
        return outputs, src_matrices, ray_directions, depth_values

    def pos_enc(self, values, pos_enc_basis, type):
        input_values = values
        if pos_enc_basis > 0:
            if self.use_pos_enc == 'fourier':
                basis_values = torch.cat(pos_enc_basis * [input_values], dim=-1)
                value = 2 * np.pi * basis_values * self.fourier_coefficients
                fin_values = torch.cat([torch.sin(value), torch.cos(value)], dim=-1)
            else:
                batch_shape = values.shape[:-1]
                scales = 2.0 ** torch.arange(0, pos_enc_basis).to(self.device)
                xb = values[..., None, :] * scales[:, None]
                four_feat = torch.sin(torch.stack([xb, xb + 0.5 * torch.pi], axis=-2))

                if self.use_pos_enc == 'windowed':
                    window = self.windowed_pos_enc(pos_enc_basis)
                    four_feat = window[..., None, None] * four_feat
                elif self.use_pos_enc == 'freq_windowed':
                    window = self.freq_mask_alpha.to(self.device)
                    four_feat = window[..., None, None] * four_feat
                
                four_feat = four_feat.reshape((*batch_shape, -1))
                fin_values = torch.cat([input_values, four_feat], dim=-1)
        else: fin_values = input_values
        return fin_values
    
    def update_freq_mask_alpha(self, current_iter, max_iter):
        # based on https://github.com/Jiawei-Yang/FreeNeRF/blob/main/internal/math.py#L277
        pos_enc_basis = self.pos_enc_basis
        if current_iter < max_iter:
            freq_mask = np.zeros(pos_enc_basis)
            ptr = (pos_enc_basis * current_iter) / max_iter
            int_ptr = int(ptr)
            freq_mask[: int_ptr + 1] = 1.0  # assign the integer part
            freq_mask[int_ptr : int_ptr + 1] = (ptr - int_ptr)  # assign the fractional part

            self.freq_mask_alpha = torch.clip(torch.from_numpy(freq_mask), 1e-8, 1-1e-8).float() # for numerical stability
            self.windowed_alpha = ptr
        else:
            self.freq_mask_alpha = torch.ones(pos_enc_basis).float()
            self.windowed_alpha = pos_enc_basis

    def save(self, filename: str, training_information: dict) -> None:
        """
        Save the CPPN model

        Args:
            filename (str): path filepath on which the model will be saved
            training_information (dict): dictionary containing information on the training
        """
        save_parameters = {
            'version': self.version,
            'parameters': self.model_definition,
            'training_information': training_information,
            'model': self.state_dict(),
        }

        if 'windowed' in self.use_pos_enc:
            save_parameters['windowed_alpha'] = self.windowed_alpha
        
        if 'freq_windowed' in self.use_pos_enc:
            save_parameters['freq_mask_alpha'] = self.freq_mask_alpha

        torch.save(
            save_parameters,
            f=filename)