import torch
import pdb

def x_rotation_matrix(angle):
    rotation = torch.eye(4).unsqueeze(dim=0).repeat_interleave(angle.size()[0], dim=0)
    rotation[:,1,1] = torch.cos(angle)
    rotation[:,1,2] = -torch.sin(angle)
    rotation[:,2,1] = torch.sin(angle)
    rotation[:,2,2] = torch.cos(angle)
    return rotation

def y_rotation_matrix(angle):
    rotation = torch.eye(4).unsqueeze(dim=0).repeat_interleave(angle.size()[0], dim=0)
    rotation[:,0,0] = torch.cos(angle)
    rotation[:,0,2] = torch.sin(angle)
    rotation[:,2,0] = -torch.sin(angle)
    rotation[:,2,2] = torch.cos(angle)
    return rotation

def z_rotation_matrix(angle):
    rotation = torch.eye(4).unsqueeze(dim=0).repeat_interleave(angle.size()[0], dim=0)
    rotation[:,0,0] = torch.cos(angle)
    rotation[:,0,1] = -torch.sin(angle)
    rotation[:,1,0] = torch.sin(angle)
    rotation[:,1,1] = torch.cos(angle)
    return rotation

def translation_matrix(vec):
    m = torch.eye(4).unsqueeze(dim=0).repeat_interleave(vec.shape[0], dim=0)
    m[:, :3, 3] = vec[:, :3]
    return m

def get_rotation(theta, phi, larm):
    R1 = z_rotation_matrix(larm)
    R2 = y_rotation_matrix(theta)
    R3 = x_rotation_matrix(phi)
    R = torch.matmul(R1, torch.matmul(R2, R3))
    return R

def get_cam_matrices_table(poses, distance_source_to_iso):
    # translate table offset
    m1 = translation_matrix(-poses[:, -3:])
    
    # rotate geometry
    m2 = get_rotation(poses[:,0], poses[:,1], poses[:,2])
    
    # translate to source
    m4 = translation_matrix(distance_source_to_iso + poses[:, 3:6])

    cam_to_worlds = m1 @ (m2 @ m4)

    return cam_to_worlds

def randomize_depth(z_vals, device):
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.concat([mids, z_vals[..., -1:]], -1)
    lower = torch.concat([z_vals[..., :1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape).to(device)
    z_vals = lower + (upper - lower) * t_rand
    depth_values = z_vals.to(device)

    return depth_values

def get_minibatches(inputs, chunksize=1024*8):
  r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
  Each element of the list (except possibly the last) has dimension `0` of length
  `chunksize`.
  """
  return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]