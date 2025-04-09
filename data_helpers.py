import numpy as np
import pandas as pd
import os
import wandb
import torch
import copy

from datetime import datetime
from ast import literal_eval

def load_data(data_name, file_name=None, binary=False, data_size=None, step_size=None, num_proj=None, combination_proj='', only_pci_vessels=True):
    unseen_ray_df = None

    image_size = data_size if data_size else 106
    data_folder_name = f'data/{data_name}/'
    store_folder_name = f'cases/{data_name}/'

    proj_file_name = f'{data_folder_name}df-{image_size}-{num_proj}'
    ray_file_name = f'{data_folder_name}df-rays-{image_size}-{num_proj}'
    if len(combination_proj) > 0:
        proj_file_name += f'-{combination_proj}'
        ray_file_name += f'-{combination_proj}'

    if not only_pci_vessels:
        proj_file_name += f'-non_pci'
        ray_file_name += f'-non_pci'
        
    proj_file_name += '.csv'
    ray_file_name += '.csv'

    print(proj_file_name)
    print(ray_file_name)
    
    proj_df = pd.read_csv(proj_file_name, sep=';')
    ray_df = pd.read_csv(ray_file_name, sep=';')

    proj_df.set_index('image_id', inplace=True)
    return proj_df, ray_df, store_folder_name, unseen_ray_df

def prepare_train_df(proj_df, ray_df, data_name, test_proj_id):
    # reformat ray origin & direction
    train_df = proj_df.copy()
    train_ray_df = ray_df.copy()
    train_ray_df['ray_origins'] = np.array([train_ray_df['ray_origins_x'].tolist(), train_ray_df['ray_origins_y'].tolist(), train_ray_df['ray_origins_z'].tolist()]).T.tolist()
    train_ray_df['ray_directions'] = np.array([train_ray_df['ray_directions_x'].tolist(), train_ray_df['ray_directions_y'].tolist(), train_ray_df['ray_directions_z'].tolist()]).T.tolist()

    train_directions = np.array([train_ray_df['directions_x'].to_numpy(), train_ray_df['directions_y'].to_numpy(), train_ray_df['directions_z'].to_numpy()]).T
    train_ray_df['directions'] = train_directions.tolist()

    train_cam2worlds = np.array(train_df['tform_cam2world'].apply(literal_eval).tolist())

    return train_df, train_ray_df, train_cam2worlds

def get_test_data(proj_df, ray_df, test_proj_id, device):
    test_ray_df = ray_df[ray_df['image_id'] == test_proj_id].copy()

    pixel_test_x_positions = np.array(test_ray_df['x_position'].tolist())
    pixel_test_y_positions = np.array(test_ray_df['y_position'].tolist())

    img_height = proj_df['org_img_height'].iloc[0]
    img_width = proj_df['org_img_width'].iloc[0]

    test_img = torch.zeros((img_height, img_width)).to(device)
    test_pix_vals = torch.from_numpy(test_ray_df['pixel_value'].to_numpy()).to(device).float()

    test_img[pixel_test_y_positions, pixel_test_x_positions] = test_pix_vals

    direction = torch.from_numpy(np.array([test_ray_df['directions_x'].to_numpy(), test_ray_df['directions_y'].to_numpy(), test_ray_df['directions_z'].to_numpy()])).T.to(device)
    direction = direction.reshape((img_width, img_height, 3))

    return test_ray_df, pixel_test_x_positions, pixel_test_y_positions, test_img, direction

def initialize_pred_imgs(train_indices, train_ray_df, img_width, img_height, nr_imgs, device):
    # initialize prediction imgs
    target_imgs = {}
    for i in range(nr_imgs):
        train_id = train_indices[i]
        case_ray_df = train_ray_df[train_ray_df['image_id'] == train_id]
        x_positions = np.array(case_ray_df['pixel_x_position'].tolist())
        y_positions = np.array(case_ray_df['pixel_y_position'].tolist())
        pixel_values = case_ray_df['pixel_value'].tolist()

        pred_img = torch.zeros((img_width, img_height))
        pred_img[x_positions, y_positions] = torch.Tensor(pixel_values).to(device)
        target_imgs[train_id] = pred_img
    
    return target_imgs

def config_parser():
    import configargparse

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path', default='nerf-for-roadmap.txt')
    parser.add_argument('--wandb_sweep_yaml', type=str, default='sweep.yaml')

    # general run info
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--x_ray_type", type=str, default='roadmap')
    parser.add_argument('--take_mask', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--data_size', type=int)
    parser.add_argument('--num_proj', type=int)
    parser.add_argument('--combination_proj', type=str)
    parser.add_argument('--only_pci_vessels', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--debug_mode', default=False, type=lambda x: (str(x).lower() == 'true'))

    # paper setting experiments
    parser.add_argument('--use_weighted_sampling', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--load_pose_shifts', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--pose_shifts_id', type=str)
    parser.add_argument('--load_pretrained_model', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--pretrained_model_path', type=str)

    # regularizers
    parser.add_argument('--cam_weight', type=float, default=0)
    parser.add_argument('--entropy_weight', type=float, default=0)
    parser.add_argument('--l1_weight', type=float, default=0)

    # model camera settings
    parser.add_argument('--learn_first_view', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--learn_iso_center', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--learn_table', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--learn_rotation', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--pos_lr', type=float, default=1e-3)

    # learn intensity
    parser.add_argument('--learn_intensity', default=False, type=lambda x: (str(x).lower() == 'true'))

    # run info
    parser.add_argument('--n_iters', type=int)
    parser.add_argument('--display_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32768)
    parser.add_argument('--early_stop_iters', type=int, default=1000)

    # models
    parser.add_argument('--num_input_channels', type=int, default=3)
    parser.add_argument('--num_output_channels', type=int, default=1)
    parser.add_argument('--nerf_num_early_layers', type=int, default=4)
    parser.add_argument('--nerf_num_late_layers', type=int, default=0)
    parser.add_argument('--nerf_num_filters', type=int, default=256)
    
    # parameters nerf
    parser.add_argument('--depth_samples_per_ray', type=int)
    parser.add_argument('--outside_thresh', type=float)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_end_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_steps', type=int, default=100000)

    parser.add_argument('--sample_mode', type=str, default='pixel')
    parser.add_argument('--sample_weights_name', type=str, default=None)
    parser.add_argument('--img_sample_size', type=int, default=64**2)
    parser.add_argument('--raw_noise_std', type=float, default=0)

    # positional encoding
    parser.add_argument('--pos_enc', type=str)
    parser.add_argument('--pos_enc_basis', type=int)
    parser.add_argument('--windowed_start', type=int, default=0)
    parser.add_argument('--windowed_stop', type=int, default=600000)
    parser.add_argument('--pos_enc_fourier_sigma', type=int)
    parser.add_argument('--pos_enc_window_decay_steps', type=int)

    # nerf acc options
    parser.add_argument('--use_nerf_acc', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--early_stop_eps", type=float)
    parser.add_argument("--alpha_thre", type=float)

    return parser

def initialize_wandb(data_name, extra_chars):
    exp_name = datetime.now().strftime("%Y-%m-%d-%H%M") + extra_chars

    wandb.init(
        project=data_name,
        notes=exp_name,
    )

    return exp_name

def overwrite_args_wandb(run_args, wandb_args):
    # we want to overwrite the args based on the sweep args
    new_args = copy.deepcopy(run_args)
    for key in wandb_args.keys():
        setattr(new_args, key, wandb_args[key])
    
    return new_args