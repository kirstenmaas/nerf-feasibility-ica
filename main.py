import pandas as pd
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

sys.path.append('.')
# import nerfacc
import wandb
import yaml
import traceback

from nerf_helpers import render_volume_density, initialize_poses
from model.model import Model
from data_helpers import load_data, config_parser, initialize_wandb, overwrite_args_wandb, get_test_data, prepare_train_df
from xray.cagtoray import cagtoray

torch.set_printoptions(precision=10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.5, 0)

def main():
    parser = config_parser()
    run_args = parser.parse_args()
    debug_mode = run_args.debug_mode

    if debug_mode:
        try:
            train()
        except Exception as e:
            print(traceback.print_exc(), file=sys.stderr)
            exit(1)
    else:
        train()

def train():
    parser = config_parser()
    run_args = parser.parse_args()

    # setup wandb
    exp_name = initialize_wandb(run_args.data_name, '-roadmap')

    # overwrite parser arsgs with wandb args if they exist
    run_args = overwrite_args_wandb(run_args, wandb.config)
    wandb.log(vars(run_args))

    # create store folder
    store_folder_name = f'cases/{run_args.data_name}/'
    log_dir = store_folder_name + 'runs/' + exp_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # prepare the data based on the hyperparameters (e.g. include all vessels, combination of projection to train with)
    number_of_projections = cagtoray(run_args.data_name, run_args.num_proj, run_args.combination_proj, run_args.only_pci_vessels)

    # load the projections and rays to train and test with
    proj_df, ray_df, _, _ = load_data(run_args.data_name, data_size=run_args.data_size, num_proj=number_of_projections, combination_proj=run_args.combination_proj, only_pci_vessels=run_args.only_pci_vessels)

    # precomputed geometry parameters for projs
    img_height = proj_df['org_img_height'].iloc[0]
    img_width = proj_df['org_img_width'].iloc[0]
    near_thresh = proj_df['near_thresh'][0]
    far_thresh = proj_df['far_thresh'][0]
    scale_factor = proj_df['scale_factor'].iloc[0]
    max_pixel_value = proj_df['max_pixel_value'].iloc[0]

    # create test df
    test_proj_id = proj_df.index[0]
    _, test_x_positions, test_y_positions, test_img, direction = get_test_data(proj_df, ray_df, test_proj_id, device)
    norm_test_img = (test_img - torch.min(test_img)) / (torch.max(test_img) - torch.min(test_img))

    pose_df = None
    # load pose shifts from another model
    if run_args.load_pose_shifts:
        pose_df = pd.read_csv(f'cases/{run_args.data_name}/runs/{run_args.pose_shifts_id}/pose-shifts.csv', sep=';')
    poses_dict, source_pts_dict, cam_dict, unshifted_cam_dict = initialize_poses(proj_df, pose_df, device)

    # create train df
    train_df, train_ray_df, _ = prepare_train_df(proj_df, ray_df, run_args.data_name, test_proj_id)

    # create a numpy array to sample from during training
    train_pixel_vals = torch.from_numpy(np.array(train_ray_df['pixel_value'])).to(device).float() #(N_img*W*H)
    train_x_pos = np.array(train_ray_df['x_position']).astype('int') #(N_img*W*H)
    train_y_pos = np.array(train_ray_df['y_position']).astype('int') #(N_img*W*H)
    train_image_ids = np.array(train_ray_df['image_id']) #(N_img*W*H)
    train_org_src_matrices = torch.stack([unshifted_cam_dict[batch_image_id].float() for batch_image_id in train_image_ids.flatten()])

    # decide which geometry is learnable
    learn_translation = run_args.learn_iso_center or run_args.learn_table
    pose_optimization = learn_translation or run_args.learn_rotation

    # precompute depth values
    depth_samples_per_ray = run_args.depth_samples_per_ray
    outside = run_args.outside_thresh
    mid_thresh = (far_thresh + near_thresh) / 2
    near_thresh = mid_thresh + outside
    far_thresh = mid_thresh - outside
    t_vals = torch.linspace(0., 1., depth_samples_per_ray)
    z_vals = near_thresh * (1.-t_vals) + far_thresh * (t_vals)
    z_vals = z_vals.to(device)

    params = {
        'num_early_layers': run_args.nerf_num_early_layers,
        'num_late_layers': run_args.nerf_num_late_layers,
        'num_filters': run_args.nerf_num_filters,
        'num_input_channels': run_args.num_input_channels,
        'num_output_channels': run_args.num_output_channels,
        'num_input_channels_views': 0,
        'use_bias': True,
        'pos_enc': run_args.pos_enc,
        'pos_enc_basis': run_args.pos_enc_basis,
        'act_func': 'relu',
        'num_train_cases': len(proj_df)-1,
        'direction': direction,
        'z_vals': z_vals,
        'depth_samples_per_ray': depth_samples_per_ray,
        'far_thresh': far_thresh,
        'near_thresh': near_thresh,
        'use_nerf_acc': run_args.use_nerf_acc,
        'early_stop_eps': run_args.early_stop_eps,
        'alpha_thre': run_args.alpha_thre,
        'poses': poses_dict,
        'source_pts': source_pts_dict,
        'scale_factor': scale_factor,
        'x_ray_type': run_args.x_ray_type,
        'pose_optimization': pose_optimization,
        'learn_iso_center': run_args.learn_iso_center,
        'learn_table': run_args.learn_table,
        'learn_rotation': run_args.learn_rotation,
        'learn_intensity': run_args.learn_intensity,
        'learn_first_view': run_args.learn_first_view,
        'chunksize': run_args.batch_size,
        'device': device,
    }

    # setup model
    model = Model(params)
    model.to(device)

    # load the pretrained model if available
    if run_args.load_pretrained_model:
        model_info = torch.load(os.path.join(run_args.pretrained_model_path, 'coarsemodel.pth'))

        # delete the keys that are not used for training currently
        keys_to_delete = []
        for key in model_info['model'].keys():
            if 'pose_shifts' in key and not key.split('pose_shifts.')[1] in poses_dict:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del model_info['model'][key]

        # add new keys in case that new poses are used
        for key in poses_dict.keys():
            if not f'pose_shifts.{key}' in model_info['model'].keys():
                model_info['model'][f'pose_shifts.{key}'] = torch.zeros(9).to(device)

        # load the trained weights
        model.load_state_dict(model_info['model'])

    model_parameters = []
    pose_parameters = []
    for name, param in model.named_parameters():
        if 'pose_shifts' in name:
            pose_parameters.append(param)
        else:
            model_parameters.append(param)

    # initialize the I_0
    test_initial_intensities = torch.stack([model.initial_intensities[id][0]*max_pixel_value for id in np.repeat(test_proj_id, (img_height*img_width))])
    batch_initial_intensities = torch.stack([model.initial_intensities[train_image_ids[0]][0]*max_pixel_value for _ in np.arange(0, run_args.img_sample_size)])

    img_optimizer = torch.optim.Adam(model_parameters, lr=run_args.lr)
    pose_optimizer = torch.optim.Adam(pose_parameters, lr=run_args.pos_lr)

    img_lr_scheduler = torch.optim.lr_scheduler.LinearLR(img_optimizer, start_factor=1, end_factor=run_args.lr_end_factor, total_iters=run_args.lr_decay_steps)
    pose_lr_scheduler = torch.optim.lr_scheduler.LinearLR(pose_optimizer, start_factor=1, end_factor=run_args.lr_end_factor, total_iters=run_args.lr_decay_steps)

    highest_psnr = -np.inf
    highest_iter = 0
    print('start training...')
    for n_iter in range(run_args.n_iters+1):
        model.train()

        # update the frequencey windowed encoding
        if run_args.pos_enc == 'freq_windowed':
            model.update_freq_mask_alpha(n_iter, run_args.pos_enc_window_decay_steps)

        # sample rays from pixels of all training images
        random_ray_ids = np.random.randint(low=0, high=train_pixel_vals.shape[0], size=(run_args.img_sample_size))
        batch_pix_vals = train_pixel_vals[random_ray_ids] # tensor
        batch_x_pos = train_x_pos[random_ray_ids] #np
        batch_y_pos = train_y_pos[random_ray_ids] #np
        batch_image_ids = train_image_ids[random_ray_ids] #py str
        org_src_matrices = train_org_src_matrices[random_ray_ids]

        # get predictions from model
        predictions, pred_src_matrices, batch_directions, batch_depth_values = model.forward_pixel_poses(batch_image_ids, batch_x_pos, batch_y_pos)
        
        unflattened_shape = list([batch_directions.shape[0], batch_depth_values.shape[0]]) + [model.num_output_channels]
        batch_pred_vals = torch.reshape(predictions, unflattened_shape)
        sc_batch_depth_values = batch_depth_values * scale_factor
        
        # render sigmas to pixels
        pix_pred_vals, _, _, entropy_loss, l1_loss = render_volume_density(batch_pred_vals, batch_initial_intensities, batch_directions, sc_batch_depth_values, run_args.take_mask)
        pixel_loss = torch.nn.functional.mse_loss(pix_pred_vals, batch_pix_vals).float()
        cam_loss = torch.nn.functional.mse_loss(pred_src_matrices, org_src_matrices)

        loss = pixel_loss + run_args.cam_weight*cam_loss + run_args.entropy_weight*entropy_loss + run_args.l1_weight*l1_loss
        psnr = -10. * torch.log10(pixel_loss)

        img_optimizer.zero_grad()
        pose_optimizer.zero_grad()

        loss.backward()

        img_optimizer.step()
        pose_optimizer.step()

        img_lr_scheduler.step()
        pose_lr_scheduler.step()

        log_dict = {
            "train_loss": loss, 
            "train_psnr": psnr,
            "train_pixel_loss": pixel_loss,
            "train_cam_loss": cam_loss,
            "train_l1_loss": l1_loss,
            "train_entropy_loss": entropy_loss,
        }

        if 'windowed' in run_args.pos_enc:
            log_dict['train_temp_windowed'] = model.windowed_alpha

        # logging
        wandb.log(log_dict)

        if n_iter % run_args.display_every == 0:
            model.eval()

            with torch.no_grad():
                org_test_src_matrix = unshifted_cam_dict[test_proj_id].float()

                predictions, pred_src_matrix, pred_ray_directions, pred_depth_values = model.forward_poses_eval(test_proj_id)

                unflattened_shape = list([pred_ray_directions.shape[0], pred_depth_values.shape[0]]) + [model.num_output_channels]
                test_pred_vals = torch.reshape(predictions, unflattened_shape)

                rd_test_depth_values = pred_depth_values * scale_factor
                
                test_pix_pred_vals, _, _, entropy_loss, l1_loss = render_volume_density(test_pred_vals, test_initial_intensities, pred_ray_directions, rd_test_depth_values, run_args.take_mask)

                test_pred_img = torch.zeros((img_height, img_width)).to(device)
                test_pred_img[test_y_positions, test_x_positions] = test_pix_pred_vals.float()
                pixel_loss = torch.nn.functional.mse_loss(test_pred_img, test_img).float()
                cam_loss = torch.nn.functional.mse_loss(pred_src_matrix, org_test_src_matrix.float())

                loss = pixel_loss + run_args.cam_weight*cam_loss + run_args.entropy_weight*entropy_loss + run_args.l1_weight*l1_loss
                psnr = -10. * torch.log10(pixel_loss)

                wandb.log({
                    "test_loss": loss, 
                    "test_psnr": psnr,
                    "test_pixel_loss": pixel_loss,
                    "test_cam_loss": cam_loss,
                    "test_entropy_loss": entropy_loss,
                    "test_l1_loss": l1_loss
                })

                if psnr >= highest_psnr:
                    highest_psnr = psnr
                    highest_iter = n_iter
                    model.save(f'{log_dir}highmodel.pth', {})

                else:
                    if highest_iter - n_iter > run_args.early_stop_iters:
                        print(f'early stop at {n_iter}!')
                        break

                norm_pred_test_img = (test_pred_img - torch.min(test_pred_img)) / (torch.max(test_pred_img) - torch.min(test_pred_img))

                wandb.log({
                    'prediction': wandb.Image(norm_pred_test_img),
                    'original': wandb.Image(norm_test_img),
                    'difference': wandb.Image(torch.abs(norm_pred_test_img-norm_test_img)),
                })

                print("Iteration:", n_iter)
                print("Loss:", loss.item())
                print("PSNR", psnr.item())

                if n_iter % run_args.save_every == 0:
                    plt.imsave(log_dir + f'coarse-proj-{test_proj_id}-{n_iter}.png', test_pred_img.cpu().numpy(), cmap='gray')
                    plt.imsave(log_dir + f'coarse-proj-{test_proj_id}-{n_iter}-diff.png', torch.abs(test_pred_img-test_img).cpu().numpy(), cmap='gray')
                    try:
                        model.save(f'{log_dir}model.pth', {})
                    except:
                        print('error saving model')

                    for i, key in enumerate(poses_dict.keys()):
                        pose_shift = torch.clone(model.pose_shifts[key])

                        pose = poses_dict[key] + pose_shift
                        print(f'{key}-{pose}-{pose_shift}')

if __name__ == "__main__":
    wandb.login()

    parser = config_parser()
    run_args = parser.parse_args()

    project_name = 'nerf-for-roadmap'

    with open(run_args.wandb_sweep_yaml, 'r') as f:
        sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)
    wandb.agent(sweep_id, function=main)