import numpy as np
from scipy.spatial.transform import Rotation as R
from collections.abc import Iterable
from helpers import *

class XRayProjection:
    def __init__(self, dicom, x_ray_type='roadmap', dicom_volume=None):
        self.dicom = dicom
        self.pixel_array = dicom.pixel_array
        self.x_ray_type = x_ray_type

        self.number_of_frames = 1
        if hasattr(dicom, 'NumberOfFrames'):
            self.number_of_frames = int(dicom.NumberOfFrames)

        self.nr_of_pixels = np.array([dicom.Columns, dicom.Rows, self.number_of_frames])
        self.max_pixel_value = 2 ** (dicom.HighBit+1) - 1

        if (0x2003, 0x20A2) in dicom:
            self.sequences = dicom[0x2003, 0x20A2]
        elif (0x2003, 0x102E) in dicom:
            self.sequences = dicom[0x2003, 0x102E]
        else:
            self.sequences = np.arange(0, self.number_of_frames)

        self.distance_source_to_detector = float(dicom.DistanceSourceToDetector)
        self.distance_source_to_patient = float(dicom.DistanceSourceToPatient)
        self.distance_detector_to_patient = self.distance_source_to_detector - self.distance_source_to_patient
        self.imager_pixel_spacing = np.array([dicom.ImagerPixelSpacing[1], dicom.ImagerPixelSpacing[0]])

        self.geo = []
        for frame_nb, element in enumerate(self.sequences):
            self.geo.append(self.RoadmapRunGeometry(element, frame_nb, dicom))

    def angles(self, idx):
        theta = np.float64(self.geo[idx].prop_angle)
        phi = np.float64(self.geo[idx].roll_angle)
        larm = np.float64(self.geo[idx].larm_angle)
        return [theta, phi, larm]
    
    def table_offset(self, idx, initial_tab_pos=[0,0,0]):
        return self.geo[idx].table_pos - initial_tab_pos
    
    def source_matrix(self, frame_idx, init_table_pos=np.array([0,0,0]), phantom=False, scale_factor=1):
        geo = self.geo[frame_idx]
        rotation = geo.rotation if not phantom else geo.phantom_rotation

        # translate table offset
        m1 = translation_matrix(-self.table_offset(frame_idx, init_table_pos)*scale_factor)
        # rotate geometry
        m2 = rotation_matrix(rotation)
        # translate to source
        m4 = translation_matrix(geo.distance_source_to_patient*scale_factor)

        mt = m1.dot(m2.dot(m4))

        return None, mt
    
    def detector_matrix(self, frame_idx, init_table_pos=np.array([0,0,0]), phantom=False, scale_factor=1):
        geo = self.geo[frame_idx]
        rotation = geo.rotation if not phantom else geo.phantom_rotation

        # translate table offset
        m1 = translation_matrix(-self.table_offset(frame_idx, init_table_pos)*scale_factor)
        # rotate geometry
        m2 = rotation_matrix(rotation)
        # translate to detector
        m3 = translation_matrix(-geo.iso_center*scale_factor)

        mt = m1.dot(m2.dot(m3))

        return None, mt
    
    class RoadmapRunGeometry:
        def __init__(self, element, frame_nb, dicom):
            # based on the dicom structure, obtained the respective angles and geometry information
            if isinstance(element, Iterable) and (0x2003, 0x20A3) in element:
                self.image_number = element[0x2003, 0x20A3].value
                self.prop_angle = np.array(element[0x2003, 0x20A4].value)
                self.roll_angle = np.array(element[0x2003, 0x20E1].value)
                self.larm_angle = np.array(element[0x2003, 0x20A5].value)

                self.iso_center = np.array(element[0x2003, 0x20A7].value)
                self.distance_source_to_detector = np.array(element[0x2003, 0x20A6].value)
                self.distance_iso_to_detector = np.array(element[0x2003, 0x20A7].value)
                self.distance_source_to_patient = self.distance_source_to_detector - self.distance_iso_to_detector
            else:
                # get the info from the global DICOM, not per frame.
                self.image_number = frame_nb
                self.prop_angle = float(dicom.PositionerPrimaryAngle)
                self.roll_angle = float(dicom.PositionerSecondaryAngle)
                self.larm_angle = 0.

                self.distance_source_to_detector = np.array([0., 0., -float(dicom.DistanceSourceToDetector)])
                self.distance_source_to_patient = np.array([0., 0., -float(dicom.DistanceSourceToPatient)])
                self.distance_detector_to_patient = self.distance_source_to_detector - self.distance_source_to_patient
                self.iso_center = self.distance_source_to_detector - self.distance_source_to_patient # no calibration

            # obtain table position per frame
            self.table_pos = TablePosFromDicom(dicom, element)

            # define rotation
            self.rotation = (
                R.from_rotvec(np.deg2rad(self.larm_angle) * np.array([0, 0, 1])) *
                R.from_rotvec(np.deg2rad(self.prop_angle) * np.array([0, 1, 0])) *
                R.from_rotvec(np.deg2rad(-self.roll_angle) * np.array([1, 0, 0])))

            self.xv_image_rotated, self.xv_flip_vertical, self.xv_flip_horizontal = False, False, False
            self.xv_detector_angle = 0.
            if isinstance(element, Iterable) and (0x2003, 0x20AE) in element:
                self.xv_image_rotated = element[0x2003, 0x20AE].value.lower() == 'true'
                self.xv_flip_vertical = element[0x2003, 0x20AA].value.lower() == 'true'
                self.xv_flip_horizontal = element[0x2003, 0x20A9].value.lower() == 'true'
                self.xv_detector_angle = element[0x2003, 0x20A8].value

def TablePosFromDicom(dicom, element):
    if isinstance(element, Iterable) and (0x2003, 0x1265) in dicom:
        AlluraBeamTransversal = dicom[0x2003, 0x1265].value
        AlluraBeamLongitudinal = dicom[0x2003, 0x1207].value
        TableTopLateralPosition = element.TableTopLateralPosition
        TableTopLongitudinalPosition = element.TableTopLongitudinalPosition
        TableTopVerticalPosition = element.TableTopVerticalPosition
        return np.array([
            TableTopLateralPosition - AlluraBeamTransversal,
            TableTopLongitudinalPosition - AlluraBeamLongitudinal,
            TableTopVerticalPosition])
    else:
        print('no table position found in dicom!')
        return np.array([0.0, 0.0, 0.0])