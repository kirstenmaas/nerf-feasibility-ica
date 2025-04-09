import torch
import numpy as np

def x_rotation_matrix(angle):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0], 
        [0, np.sin(angle), np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def y_rotation_matrix(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def z_rotation_matrix(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle), np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def source_matrix(source_pt, theta, phi, larm=0):
    R_c = get_rotation(phi, theta, larm)
    C = R_c.dot(source_pt)

    worldtocam = np.identity(4)
    worldtocam[:3, :3] = R_c[:3, :3]
    worldtocam[:3, -1] = C[:3]

    return worldtocam

def get_rotation(phi, theta, larm):
    return np.linalg.inv(z_rotation_matrix(np.deg2rad(larm)).dot(x_rotation_matrix(np.deg2rad(phi)).dot(y_rotation_matrix(np.deg2rad(theta)))))

def rotation_matrix(rotation):
    m = np.identity(4)
    m[:3, :3] = rotation.as_matrix()
    return m

def translation_matrix(vec):
    m = np.identity(4)
    m[:3, 3] = vec
    return m

def get_query_points(n_detector, resized_detector, d_detector, DSD, tform_cam2world, depth_samples_per_ray, near_thresh, far_thresh, device, randomize=False):
    # cast rays based on the geometry of the C-arm system
    ii, jj = torch.meshgrid(
        torch.linspace(0, resized_detector[0] - 1, n_detector[0]).to(device), #W
        torch.linspace(resized_detector[1] - 1, 0, n_detector[1]).to(device), #H
        indexing='ij'
    )

    uu = (ii - (resized_detector[0] - 1)/2)*d_detector[0]
    vv = (jj - (resized_detector[1] - 1)/2)*d_detector[1]
    
    dirs = torch.stack([uu / DSD, vv / DSD, -torch.ones_like(uu)], -1)
    ray_directions = torch.sum(dirs[..., None, :] * tform_cam2world[:3, :3], dim=-1).to(device)
    ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)

    t_vals = torch.linspace(0., 1., depth_samples_per_ray)
    z_vals = near_thresh * (1.-t_vals) + far_thresh * (t_vals)

    if randomize:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.concat([mids, z_vals[..., -1:]], -1)
        lower = torch.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    depth_values = z_vals.to(ray_origins.device)

    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
    return query_points, ray_origins, ray_directions, depth_values, dirs