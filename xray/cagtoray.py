import sys

sys.path.append('.')

import pydicom as pyd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import torch
from scipy.interpolate import interp2d

try:
    from xray import XRayProjection
except Exception as e:
    from .xray import XRayProjection

try:
    from helpers import get_query_points
except Exception as e:
    from .helpers import get_query_points


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cagtoray(patient_number, number_of_projections=4, combination_proj='', only_pci_vessels=True):
    patient_number = patient_number.split('-LCA')[0]
    focus_artery = 'LCA'
    x_ray_type = 'roadmap'
    take_mask = True # for learning segmentations
    save_df = True
    new_img_width = 512

    load_pose_shifts = False
    pose_shifts_id = ''

    data_name = f'{patient_number}-{focus_artery}' #-mask
    if take_mask:
        data_name += '-mask'

    # load pose shifts from another model
    if load_pose_shifts:
        pose_df = pd.read_csv(f'{pose_shifts_id}pose-shifts.csv', sep=';')
    
    # data path should be defined
    artery_folder = f'dicompath' 

    dicom_objs = []
    dicom_file_names = []
    for dir_path, dir_names, file_names in os.walk(artery_folder):
        for file_name in [f for f in file_names if f.endswith(".dcm")]:
            dicom_obj = {}
            
            dicom_file_path = f'dicompath/{file_name}'
            frame_nb = int(dir_path.split('\\')[-1])

            # only keep one frame per frame
            file_name = file_name.split('.dcm')[0]
            
            translation = [0,0,0]
            if load_pose_shifts:
                pose = pose_df[pose_df['key'].str.contains(f"{file_name}-{frame_nb}")].iloc[0]
                translation = [pose['x_table'], pose['y_table'], pose['z_table']]

            # check if obj already seen
            same_file_name_objs = [x for x in dicom_objs if x['file_name'] == file_name]
            if len(same_file_name_objs) > 0:
                dicom_obj = same_file_name_objs[0]
                obj_idx = dicom_objs.index(dicom_obj)

                existing_frame_nbs = dicom_obj['frame_nb']
                existing_frame_nbs.append(frame_nb)

                existing_paths = dicom_obj['path']
                existing_paths.append(dir_path)

                existing_translation = dicom_obj['translation']
                existing_translation.append(translation)
            else:
                dicom_obj['file_name'] = file_name
                dicom_obj['file_path'] = dicom_file_path
                dicom_obj['path'] = [dir_path]
                dicom_obj['frame_nb'] = [frame_nb]
                dicom_obj['translation'] = [translation]
                dicom_objs.append(dicom_obj)

            dicom_file_names.append(file_name)

    depth_samples_per_ray = 50
    thresh_outside = 125

    folder_name = f'data/{data_name}/'
    folder_img_name = f'{folder_name}{new_img_width}/'

    # create folder for data
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    # create folder for imgs
    if not os.path.isdir(folder_img_name):
        os.mkdir(folder_img_name)

    image_ids = []
    thetas = []
    phis = []
    larms = []
    translations_x = []
    translations_y = []
    translations_z = []

    matrices = []
    det_matrices = []
    images = []

    img_widths = []
    img_heights = []
    focal_lengths = []
    imager_pixel_spacings = []
    near_threshs = []
    far_threshs = []
    depth_samples = []
    depth_values_lst = []
    scale_factors = []
    src_pts = []
    max_pixel_values = []

    pixel_image_ids = []
    x_positions = []
    y_positions = []
    pixel_values = []

    ray_origins_x = []
    ray_origins_y = []
    ray_origins_z = []

    ray_directions_x = []
    ray_directions_y = []
    ray_directions_z = []

    directions_x = []
    directions_y = []
    directions_z = []

    init_table_pos = [0,0,0]
    scale_factor = 1e-2
    volume_scale_factor = 5e-3

    # manually define which frames to use and which table position to consider
    frames_to_keep = {}
    table_pos_to_use = {}
    
    new_dicom_objs = []
    for obj in dicom_objs:
        if obj['file_name'] in frames_to_keep.keys():
            new_dicom_objs.append(obj)
    dicom_objs = np.array(new_dicom_objs)

    # number of projections cannot be larger than the number of dicom objs
    number_of_projections = min(len(dicom_objs), number_of_projections)

    # if a combination is given, iterate over that combination specifically
    comb_ind = []
    if len(combination_proj) > 0:
        if len(combination_proj) == number_of_projections:
            comb_ind = np.array([*combination_proj]).astype('int')
            iter_dicom_objs = dicom_objs[comb_ind]
            print(f'using combinations {comb_ind}')
        else:
            print('combination does not match length of number of projections')
            return ValueError()
    else:
        iter_dicom_objs = dicom_objs[:number_of_projections]

    for i, dicom_obj in enumerate(iter_dicom_objs):
        dicom_proj = pyd.read_file(dicom_obj['file_path'])

        print(dicom_obj['file_name'])

        # list of frame ids that are annotated
        frame_ids_projs = dicom_obj['frame_nb']
        
        pixel_array = dicom_proj.pixel_array

        if dicom_obj['file_name'] not in frames_to_keep:
            print(f"Skipped {dicom_obj['file_name']}")
            continue

        if len(pixel_array.shape) == 2:
            pixel_array = np.expand_dims(pixel_array, 0)

        nb_frames = pixel_array.shape[0]
        frame_to_keep = frames_to_keep[dicom_obj['file_name']]
        img_height, img_width = pixel_array.shape[1:]

        x_ray_run = XRayProjection(dicom_proj, x_ray_type)
        
        ratio = img_height / img_width
        imager_pixel_spacing = x_ray_run.imager_pixel_spacing

        imager_pixel_spacing = imager_pixel_spacing[::-1]*volume_scale_factor

        if i == 0:
            init_table_pos = x_ray_run.geo[0].table_pos

        for frame_idx in frame_ids_projs:
            image_geo = x_ray_run.geo[frame_idx]
            theta, phi, larm = x_ray_run.angles(frame_idx)
            image_id = f"{dicom_obj['file_name']}-{frame_idx}-{int(theta)}-{int(phi)}"

            # it is not the frame to keep
            if frame_idx != frame_to_keep:
                continue

            # we have seen the frame before...
            if image_id in image_ids:
                continue

            print(frame_idx)

            dicom_obj_lst_id = frame_ids_projs.index(frame_idx)

            focal_length = -x_ray_run.distance_source_to_detector*volume_scale_factor
            iso_center = image_geo.iso_center

            img = pixel_array[frame_idx]

            if image_geo.xv_flip_horizontal:
                img = np.fliplr(img)
            if image_geo.xv_flip_vertical:
                img = np.flipud(img)

            # only take the mask
            if take_mask:
                mask_file_name = 'primary_mask'
                if not only_pci_vessels:
                    mask_file_name = 'secondary_mask'
                mask_img = mpimg.imread(f"{dicom_obj['path'][dicom_obj_lst_id]}/{mask_file_name}.png")[:, :, 0] # grayscale
                
                # make sure that background is black and foreground is white
                unique_counts = np.unique(mask_img, return_counts=True)[1]

                # binarize the image
                if len(unique_counts != 2):
                    mask_img[mask_img < 0.5] = 0
                    mask_img[mask_img > 0.5] = 1
                    unique_counts = np.unique(mask_img, return_counts=True)[1]

                black_counts, white_counts = unique_counts

                if white_counts > black_counts:
                    mask_img = np.abs(mask_img - 1)

                img = mask_img.astype('int')

            # for our data, we need to negate the phi angle
            phi = -phi

            # use the table position as initialization for the model
            frame_table_pos = init_table_pos
            if not table_pos_to_use[dicom_obj['file_name']]:
                print('not using the DICOM table position!')
                frame_table_pos = x_ray_run.geo[0].table_pos

            _, source_extr = x_ray_run.source_matrix(frame_idx, frame_table_pos, phantom=False, scale_factor=volume_scale_factor)
            _, det_extr = x_ray_run.detector_matrix(frame_idx, frame_table_pos, phantom=False, scale_factor=volume_scale_factor)

            src_pt = np.array([iso_center[0], iso_center[1], -(x_ray_run.distance_source_to_detector + iso_center[2])])*volume_scale_factor

            # table_offset = table_pos
            table_offset = x_ray_run.table_offset(frame_idx, frame_table_pos)*volume_scale_factor
            sample_iso_center = -(x_ray_run.distance_source_to_detector + iso_center[2])*volume_scale_factor

            near_thresh = sample_iso_center + thresh_outside
            far_thresh = sample_iso_center - thresh_outside

            x = np.arange(img_width)
            y = np.arange(img_height)
            f_interp = interp2d(x, y, img, kind="cubic")

            # resize image
            new_img_height = int(ratio * new_img_width)

            detector_size = np.array([img_width, img_height])
            n_detector = np.array([new_img_width, new_img_height])
            nb_pixels = new_img_width * new_img_height

            # evenly sample across image to obtain new image
            ii_p, jj_p = np.meshgrid(
                np.linspace(0, detector_size[0] - 1, n_detector[0]), #W
                np.linspace(0, detector_size[1] - 1, n_detector[1]), #H
                indexing='ij'
            )
            
            # only important when the image size is non-int (i.e. resized_detector != n_detector)
            pos_x = np.unique(ii_p) / np.max(ii_p) * (img.shape[1]-1)
            pos_y = np.unique(jj_p) / np.max(jj_p) * (img.shape[0]-1)
            img = f_interp(pos_x, pos_y)

            max_pixel_value = x_ray_run.max_pixel_value
            if take_mask:
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img[img < 0.5] = 0
                img[img >= 0.5] = 1
            else:
                # remove the negative values (mostly the edges of the image)
                img = np.clip(img, a_min=1, a_max=max_pixel_value)
                img = np.log10(img)
                max_pixel_value = np.log10(max_pixel_value)

            # pixel positions
            ii, jj = np.meshgrid(
                np.linspace(0, n_detector[0] - 1, n_detector[0]), #W
                np.linspace(0, n_detector[1] - 1, n_detector[1]), #H
                indexing='ij'
            )

            tform_cam2world = torch.from_numpy(source_extr).to(device)
            query_points, ray_origins, ray_directions, depth_values, directions = get_query_points(
                n_detector, detector_size, imager_pixel_spacing, focal_length, tform_cam2world, 
                depth_samples_per_ray, near_thresh, far_thresh, 
                device, randomize=False)

            plt.imsave(f'{folder_img_name}{image_id}.png', img, cmap='gray')

            image_ids.append(image_id)
            thetas.append(theta)
            phis.append(phi)
            larms.append(larm)
            translations_x.append(table_offset[0])
            translations_y.append(table_offset[1])
            translations_z.append(table_offset[2])
            matrices.append(source_extr.tolist())
            det_matrices.append(det_extr.tolist())
            images.append(img.tolist())
            img_widths.append(img.shape[1])
            img_heights.append(img.shape[0])
            focal_lengths.append(focal_length)
            imager_pixel_spacings.append(imager_pixel_spacing)
            near_threshs.append(near_thresh)
            far_threshs.append(far_thresh)
            scale_factors.append(scale_factor)
            depth_samples.append(depth_samples_per_ray)
            depth_values_lst.append(depth_values.tolist())
            src_pts.append(src_pt)
            max_pixel_values.append(max_pixel_value)

            pixel_image_ids.append(np.repeat(image_id, nb_pixels))
            x_positions.append(ii)
            y_positions.append(jj)
            
            pixel_values.append(img.T.flatten())

            ray_origins = ray_origins.reshape((-1, 3)).cpu().numpy()
            ray_origins_x.append(ray_origins[:,0])
            ray_origins_y.append(ray_origins[:,1])
            ray_origins_z.append(ray_origins[:,2])

            ray_directions = ray_directions.reshape((-1,3)).cpu().numpy()
            ray_directions_x.append(ray_directions[:,0])
            ray_directions_y.append(ray_directions[:,1])
            ray_directions_z.append(ray_directions[:,2])

            directions = directions.reshape((-1, 3)).cpu().numpy()
            directions_x.append(directions[:,0])
            directions_y.append(directions[:,1])
            directions_z.append(directions[:,2])

            # only consider one frame per angiogram projection
            continue

    if take_mask:
        # normalize the pixel values
        pixel_values = (pixel_values - np.min(pixel_values)) /  (np.max(pixel_values) - np.min(pixel_values))

    proj_df = pd.DataFrame({'image_id': image_ids, 'theta': thetas, 'phi': phis, 'larm': larms, 'translation_x': translations_x,
            'translation_y': translations_y, 'translation_z': translations_z, 'tform_cam2world': matrices, 'image_data': images,
            'org_img_width': img_widths, 'org_img_height': img_heights, 'focal_length': focal_lengths,
            'near_thresh': near_threshs, 'far_thresh': far_threshs, 'scale_factor': scale_factors, 'imager_pixel_spacing': imager_pixel_spacings,
            'depth_sample': depth_samples, 'depth_values': depth_values_lst, 'src_pt': src_pts, 'max_pixel_value': max_pixel_values })

    proj_df['imager_pixel_spacing'] = proj_df['imager_pixel_spacing'].map(list)
    proj_df['image_data'] = proj_df['image_data'].map(list)
    proj_df['tform_cam2world'] = proj_df['tform_cam2world'].map(list)
    proj_df['src_pt'] = proj_df['src_pt'].map(list)

    proj_file_name = f'{folder_name}/df-{new_img_width}-{number_of_projections}'

    if len(combination_proj) > 0:
        proj_file_name += f'-{combination_proj}'

    if not only_pci_vessels:
        proj_file_name += f'-non_pci'

    proj_file_name += '.csv'

    if save_df:
        proj_df.to_csv(proj_file_name, sep=';')
        print('saved proj df...')  

    pixel_image_ids = np.array(pixel_image_ids).reshape(-1).tolist()
    pixel_values = np.array(pixel_values).reshape(-1).tolist()

    x_positions = np.array(x_positions).reshape(-1).tolist()
    y_positions = np.array(y_positions).reshape(-1).tolist()

    ray_origins_x = np.array(ray_origins_x).reshape(-1).tolist()
    ray_origins_y = np.array(ray_origins_y).reshape(-1).tolist()
    ray_origins_z = np.array(ray_origins_z).reshape(-1).tolist()

    ray_directions_x = np.array(ray_directions_x).reshape(-1).tolist()
    ray_directions_y = np.array(ray_directions_y).reshape(-1).tolist()
    ray_directions_z = np.array(ray_directions_z).reshape(-1).tolist()

    directions_x = np.array(directions_x).reshape(-1).tolist()
    directions_y = np.array(directions_y).reshape(-1).tolist()
    directions_z = np.array(directions_z).reshape(-1).tolist()

    ray_df = pd.DataFrame({'image_id': pixel_image_ids, 'pixel_value': pixel_values, 'x_position': x_positions, 'y_position': y_positions, 
        'ray_origins_x': ray_origins_x, 'ray_origins_y': ray_origins_y, 'ray_origins_z': ray_origins_z,
        'ray_directions_x': ray_directions_x, 'ray_directions_y': ray_directions_y, 'ray_directions_z': ray_directions_z,
        'directions_x': directions_x, 'directions_y': directions_y, 'directions_z': directions_z })

    ray_file_name = f'{folder_name}/df-rays-{new_img_width}-{number_of_projections}'

    if len(combination_proj) > 0:
        ray_file_name += f'-{combination_proj}'

    if not only_pci_vessels:
        ray_file_name += f'-non_pci'

    ray_file_name += '.csv'

    if save_df:
        ray_df.to_csv(ray_file_name, sep=';')
        print('saved ray df...')  

    return number_of_projections
            
if __name__ == '__main__':
    patient_number = 'Medusa-annotate-00003'
    number_of_projections = 4
    combination_proj = ''
    cagtoray(patient_number, number_of_projections, combination_proj=combination_proj)