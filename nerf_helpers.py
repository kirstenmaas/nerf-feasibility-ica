import torch
import numpy as np
from ast import literal_eval

def initialize_poses(proj_df, pose_df, device):
    # initialize poses with 6 dof pose: theta, phi, larm, with shift (isocenter): x, y, z
    # model needs to learn the shift, and receives the "unshifted" version of the parameter
    poses = []
    source_pts = []

    indices = proj_df.index.tolist()
    for _, proj_idx in enumerate(indices):
        case_proj_df = proj_df[proj_df.index == proj_idx]
        pose = np.array([np.deg2rad(case_proj_df['theta'][0]), np.deg2rad(case_proj_df['phi'][0]), np.deg2rad(case_proj_df['larm'][0]), 0, 0, 0, case_proj_df['translation_x'][0], case_proj_df['translation_y'][0], case_proj_df['translation_z'][0]])

        # we add the learned shifts (from another model) to the initialized pose of the new model 
        if pose_df:
            pose_row = pose_df[pose_df['key'] == proj_idx].iloc[0]
            pose_values = pose_row.values.tolist()
            pose = pose + pose_values[2:]

        source_pt = literal_eval(case_proj_df['src_pt'].iloc[0])

        poses.append(pose)
        source_pts.append(source_pt)
    poses = torch.from_numpy(np.array(poses)).to(device)
    source_pts = torch.from_numpy(np.array(source_pts)).to(device)

    poses_dict = dict(zip(indices, poses))
    source_pts_dict = dict(zip(indices, source_pts))

    cam2worlds = np.array(proj_df['tform_cam2world'].apply(literal_eval).tolist())
    # src_matrix model starts with (excluding shifts)
    unshifted_cam2worlds = cam2worlds
    # ground truth camera (with translations and rotational shifts)
    cam_dict = dict(zip(indices, torch.from_numpy(cam2worlds).to(device)))
    # camera (without translations and rotational shifts)
    unshifted_cam_dict = dict(zip(indices, torch.from_numpy(unshifted_cam2worlds).to(device)))

    return poses_dict, source_pts_dict, cam_dict, unshifted_cam_dict

def render_volume_density(radiance_field, initial_intensities, ray_directions, depth_values, take_mask=False):
  # maps radiance field to pixel intensities / pixel mask

  one_e_10 = torch.tensor([1e-10], dtype=ray_directions.dtype, device=ray_directions.device)

  depth_values = torch.abs(depth_values)
  dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], one_e_10.expand(depth_values[..., :1].shape)), dim=-1)

  # Multiply each distance by the norm of its corresponding direction ray
  # to convert to real world distance (accounts for non-unit directions).
  norm_dists = dists * torch.norm(ray_directions[..., None, :], dim=-1)
  sigma_a = torch.nn.Softplus()(radiance_field[...,-1])
  weights = sigma_a * norm_dists

  if not take_mask:
    # model it through the beer-lambert law
    rgb_map = initial_intensities - torch.sum(weights, dim=-1)
  else:
    # model it as a maximum intensity projection
    rgb_map = torch.max(sigma_a, dim=-1).values

  depth_map = rgb_map

  entropy = get_ray_entropy(sigma_a, norm_dists)
  l1_loss = torch.sum(sigma_a, dim=-1).mean()

  return rgb_map, depth_map, weights, entropy, l1_loss

def get_ray_entropy(sigmas, dists, mask_threshold=1e-10):
  # calculates ray entropy given a mask threshold
  
  sigmas_sum = torch.sum(sigmas, dim=-1, keepdim=True)
  sigmas_sum_frac = sigmas_sum / sigmas.shape[-1]

  # disregard the rays that don't hit the object (based on a set threshold)
  mask = torch.where(sigmas_sum_frac > mask_threshold, 1, 0)
  p = torch.clip(sigmas / torch.clip(sigmas_sum, min=1e-15), min=1e-15)

  entropy = mask * -torch.sum(p * torch.log(p), dim=-1, keepdim=True)
  entropy = entropy.mean()

  return entropy