import os
from collections import OrderedDict
import cv2
import h5py
import torch
import numpy as np
from functools import partial
from torch.utils.data import Dataset
from PIL import Image
import random
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.camera_utils import load_camera_data, load_intrinsic_DAIR_V2X
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import tfm_to_pose, rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import veh_side_rot_and_trans_to_trasnformation_matrix
from opencood.utils.transformation_utils import inf_side_rot_and_trans_to_trasnformation_matrix
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor

<<<<<<< HEAD
def build_idx_to_info(data):
    idx2info = {}
    for elem in data:
        if elem["pointcloud_path"] == "":
            continue
        idx = elem["pointcloud_path"].split("/")[-1].replace(".pcd", "")
        idx2info[idx] = elem
    return idx2info

def build_inf_fid_to_veh_fid(data):
    inf_fid2veh_fid = {}
    for elem in data:
        veh_fid = elem["vehicle_pointcloud_path"].split("/")[-1].rstrip('.pcd') # "vehicle_pointcloud_path": "vehicle-side/velodyne/006206.pcd"
        inf_fid = elem["infrastructure_pointcloud_path"].split("/")[-1].rstrip('.pcd') # "infrastructure_pointcloud_path": "infrastructure-side/velodyne/013902.pcd"
        inf_fid2veh_fid[inf_fid] = veh_fid
    return inf_fid2veh_fid


def id_to_str(id, digits=6):
    result = ""
    for i in range(digits):
        result = str(id % 10) + result
        id //= 10
    return result

=======
>>>>>>> origin/lsf
class DAIRV2XBaseDataset(Dataset):
    def __init__(self, params, visualize, train=True):
        self.params = params
        self.visualize = visualize
        self.train = train
<<<<<<< HEAD
        
        if 'time_delay' in params and params['time_delay']:
            self.time_delay = params['time_delay']
        else:
            self.time_delay = 0
        print(f'*** time_delay = {self.time_delay} ***')
        
=======

>>>>>>> origin/lsf
        self.pre_processor = build_preprocessor(params["preprocess"], train)
        self.post_processor = build_postprocessor(params["postprocess"], train)
        self.post_processor.generate_gt_bbx = self.post_processor.generate_gt_bbx_by_iou
        if 'data_augment' in params: # late and early
            self.data_augmentor = DataAugmentor(params['data_augment'], train)
        else: # intermediate
            self.data_augmentor = None

        if 'clip_pc' in params['fusion']['args'] and params['fusion']['args']['clip_pc']:
            self.clip_pc = True
        else:
            self.clip_pc = False

        if 'train_params' not in params or 'max_cav' not in params['train_params']:
            self.max_cav = 2
        else:
            self.max_cav = params['train_params']['max_cav']

        self.load_lidar_file = True if 'lidar' in params['input_source'] or self.visualize else False
<<<<<<< HEAD
        # self.load_camera_file = True if 'camera' in params['input_source'] else False
        # self.load_depth_file = True if 'depth' in params['input_source'] else False
        self.load_camera_file = False
        self.load_depth_file = False
=======
        self.load_camera_file = True if 'camera' in params['input_source'] else False
        self.load_depth_file = True if 'depth' in params['input_source'] else False

>>>>>>> origin/lsf
        assert self.load_depth_file is False

        self.label_type = params['label_type'] # 'lidar' or 'camera'
        self.generate_object_center = self.generate_object_center_lidar if self.label_type == "lidar" \
                                                    else self.generate_object_center_camera

        if self.load_camera_file:
            self.data_aug_conf = params["fusion"]["args"]["data_aug_conf"]

        if self.train:
            split_dir = params['root_dir']
        else:
            split_dir = params['validate_dir']

        self.root_dir = params['data_dir']
<<<<<<< HEAD
=======

>>>>>>> origin/lsf
        self.split_info = read_json(split_dir)
        co_datainfo = read_json(os.path.join(self.root_dir, 'cooperative/data_info.json'))
        self.co_data = OrderedDict()
        for frame_info in co_datainfo:
            veh_frame_id = frame_info['vehicle_image_path'].split("/")[-1].replace(".jpg", "")
            self.co_data[veh_frame_id] = frame_info

<<<<<<< HEAD
        self.inf_fid2veh_fid = build_inf_fid_to_veh_fid(read_json(os.path.join(self.root_dir, "cooperative/data_info.json"))
        )

        self.inf_idx2info = build_idx_to_info(
            read_json(os.path.join(self.root_dir, "infrastructure-side/data_info.json"))
        )

        self.data = []
        for veh_idx in self.split_info:
            if self.is_valid_id(veh_idx):
                self.data.append(veh_idx)
        print(len(self.data))
=======
>>>>>>> origin/lsf
        if "noise_setting" not in self.params:
            self.params['noise_setting'] = OrderedDict()
            self.params['noise_setting']['add_noise'] = False
    
    def reinitialize(self):
        pass
<<<<<<< HEAD
    
    def is_valid_id(self, veh_frame_id):
        """
        Written by sizhewei @ 2022/10/05
        Given veh_frame_id, determine whether there is a corresponding inf_frame that meets the k delay requirement.

        Parameters
        ----------
        veh_frame_id : 05d
            Vehicle frame id

        Returns
        -------
        bool valud
            True means there is a corresponding road-side frame.
        """
        # print('veh_frame_id: ',veh_frame_id,'\n')
        frame_info = {}
        
        frame_info = self.co_data[veh_frame_id]
        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
        cur_inf_info = self.inf_idx2info[inf_frame_id]
        if (int(inf_frame_id) - self.time_delay < int(cur_inf_info["batch_start_id"])):
            print('short skip')
            return False
        # for i in range(self.time_delay+1):# TODO:
        delay_id = id_to_str(int(inf_frame_id) - self.time_delay)
        if delay_id not in self.inf_fid2veh_fid.keys():
            print('skip')
            return False

        return True
=======
>>>>>>> origin/lsf

    def retrieve_base_data(self, idx):
        """
        Given the index, return the corresponding data.
        NOTICE!
        It is different from Intermediate Fusion and Early Fusion
        Label is not cooperative and loaded for both veh side and inf side.
        Parameters
        ----------
        idx : int
            Index given by dataloader.
        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
<<<<<<< HEAD
        # veh_frame_id = self.split_info[idx]
        veh_frame_id = self.data[idx]
=======
        veh_frame_id = self.split_info[idx]
>>>>>>> origin/lsf
        frame_info = self.co_data[veh_frame_id]
        system_error_offset = frame_info["system_error_offset"]
        data = OrderedDict()

        data[0] = OrderedDict()
        data[0]['ego'] = True
        data[1] = OrderedDict()
        data[1]['ego'] = False

        data[0]['params'] = OrderedDict()
        data[1]['params'] = OrderedDict()
        
        # pose of agent 
        lidar_to_novatel = read_json(os.path.join(self.root_dir,'vehicle-side/calib/lidar_to_novatel/'+str(veh_frame_id)+'.json'))
        novatel_to_world = read_json(os.path.join(self.root_dir,'vehicle-side/calib/novatel_to_world/'+str(veh_frame_id)+'.json'))
        transformation_matrix = veh_side_rot_and_trans_to_trasnformation_matrix(lidar_to_novatel, novatel_to_world)
        data[0]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        inf_frame_id = frame_info['infrastructure_image_path'].split("/")[-1].replace(".jpg", "")
<<<<<<< HEAD
        delay_id = id_to_str(int(inf_frame_id) - self.time_delay)
        veh_frame_id_cur = self.inf_fid2veh_fid[delay_id]
        virtuallidar_to_world = read_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(delay_id)+'.json'))
=======
        virtuallidar_to_world = read_json(os.path.join(self.root_dir,'infrastructure-side/calib/virtuallidar_to_world/'+str(inf_frame_id)+'.json'))
>>>>>>> origin/lsf
        transformation_matrix = inf_side_rot_and_trans_to_trasnformation_matrix(virtuallidar_to_world, system_error_offset)
        data[1]['params']['lidar_pose'] = tfm_to_pose(transformation_matrix)

        data[0]['params']['vehicles_front'] = read_json(os.path.join(self.root_dir,frame_info['cooperative_label_path'].replace("label_world", "label_world_backup"))) 
        data[0]['params']['vehicles_all'] = read_json(os.path.join(self.root_dir,frame_info['cooperative_label_path'])) 

        data[1]['params']['vehicles_front'] = [] # we only load cooperative label in vehicle side
        data[1]['params']['vehicles_all'] = [] # we only load cooperative label in vehicle side
<<<<<<< HEAD
        delay_frame_info = self.co_data[veh_frame_id_cur]
=======
>>>>>>> origin/lsf

        if self.load_camera_file:
            data[0]['camera_data'] = load_camera_data([os.path.join(self.root_dir, frame_info["vehicle_image_path"])])
            data[0]['params']['camera0'] = OrderedDict()
            data[0]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                            read_json(os.path.join(self.root_dir, 'vehicle-side/calib/lidar_to_camera/'+str(veh_frame_id)+'.json')))
            data[0]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                            read_json(os.path.join(self.root_dir, 'vehicle-side/calib/camera_intrinsic/'+str(veh_frame_id)+'.json')))
            
<<<<<<< HEAD
            data[1]['camera_data']= load_camera_data([os.path.join(self.root_dir,delay_frame_info["infrastructure_image_path"])])
            data[1]['params']['camera0'] = OrderedDict()
            data[1]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                            read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/virtuallidar_to_camera/'+str(veh_frame_id_cur)+'.json')))
            data[1]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                            read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/camera_intrinsic/'+str(veh_frame_id_cur)+'.json')))

        
        if self.load_lidar_file or self.visualize:
            data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
            data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,delay_frame_info["infrastructure_pointcloud_path"]))
=======
            data[1]['camera_data']= load_camera_data([os.path.join(self.root_dir,frame_info["infrastructure_image_path"])])
            data[1]['params']['camera0'] = OrderedDict()
            data[1]['params']['camera0']['extrinsic'] = rot_and_trans_to_trasnformation_matrix( \
                                            read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/virtuallidar_to_camera/'+str(inf_frame_id)+'.json')))
            data[1]['params']['camera0']['intrinsic'] = load_intrinsic_DAIR_V2X( \
                                            read_json(os.path.join(self.root_dir, 'infrastructure-side/calib/camera_intrinsic/'+str(inf_frame_id)+'.json')))


        if self.load_lidar_file or self.visualize:
            data[0]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["vehicle_pointcloud_path"]))
            data[1]['lidar_np'], _ = pcd_utils.read_pcd(os.path.join(self.root_dir,frame_info["infrastructure_pointcloud_path"]))
>>>>>>> origin/lsf


        # Label for single side
        data[0]['params']['vehicles_single_front'] = read_json(os.path.join(self.root_dir, \
                                'vehicle-side/label/lidar_backup/{}.json'.format(veh_frame_id)))
        data[0]['params']['vehicles_single_all'] = read_json(os.path.join(self.root_dir, \
                                'vehicle-side/label/lidar/{}.json'.format(veh_frame_id)))
        data[1]['params']['vehicles_single_front'] = read_json(os.path.join(self.root_dir, \
                                'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id)))
        data[1]['params']['vehicles_single_all'] = read_json(os.path.join(self.root_dir, \
                                'infrastructure-side/label/virtuallidar/{}.json'.format(inf_frame_id)))
<<<<<<< HEAD
=======

>>>>>>> origin/lsf
        if getattr(self, "heterogeneous", False):
            self.generate_object_center_lidar = \
                                partial(self.generate_object_center_single_hetero, modality='lidar')
            self.generate_object_center_camera = \
                                partial(self.generate_object_center_single_hetero, modality='camera')

            # by default
            data[0]['modality_name'] = 'm1'
            data[1]['modality_name'] = 'm2'
            

            if self.train: # randomly choose LiDAR or Camera to be Ego
                p = np.random.rand()
                if p > 0.5:
                    data[0], data[1] = data[1], data[0]
                    data[0]['ego'] = True
                    data[1]['ego'] = False
            else:
                # evaluate, the agent of ego modality should be ego
                assert self.adaptor.mapping_dict[data[0]['modality_name']] in self.ego_modality
                

            data[0]['modality_name'] = self.adaptor.reassign_cav_modality(data[0]['modality_name'], 0)
            data[1]['modality_name'] = self.adaptor.reassign_cav_modality(data[1]['modality_name'], 1)
            

        return data


    def __len__(self):
<<<<<<< HEAD
        return len(self.data)
        # return len(self.split_info)
=======
        return len(self.split_info)
>>>>>>> origin/lsf

    def __getitem__(self, idx):
        pass


    def generate_object_center_lidar(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_all']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)

    def generate_object_center_camera(self,
                               cav_contents,
                               reference_lidar_pose):
        """
        reference lidar 's coordinate 
        """
        for cav_content in cav_contents:
            cav_content['params']['vehicles'] = cav_content['params']['vehicles_front']
        return self.post_processor.generate_object_center_dairv2x(cav_contents,
                                                        reference_lidar_pose)
                                                        
    ### Add new func for single side
    def generate_object_center_single(self,
                               cav_contents,
                               reference_lidar_pose,
                               **kwargs):
        """
<<<<<<< HEAD
        veh or inf 's coordinate
=======
        veh or inf 's coordinate. 

        reference_lidar_pose is of no use.
>>>>>>> origin/lsf
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if self.label_type == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single(cav_contents, suffix)
<<<<<<< HEAD
    
=======

>>>>>>> origin/lsf
    ### Add for heterogeneous, transforming the single label from self coord. to ego coord.
    def generate_object_center_single_hetero(self,
                                            cav_contents,
                                            reference_lidar_pose, 
                                            modality):
        """
        loading the object from single agent. 
        
        The same as *generate_object_center_single*, but it will transform the object to reference(ego) coordinate,
        using reference_lidar_pose.
        """
        suffix = "_single"
        for cav_content in cav_contents:
            cav_content['params']['vehicles_single'] = \
                    cav_content['params']['vehicles_single_front'] if modality == 'camera' else \
                    cav_content['params']['vehicles_single_all']
        return self.post_processor.generate_object_center_dairv2x_single_hetero(cav_contents, reference_lidar_pose, suffix)
<<<<<<< HEAD
        
=======


>>>>>>> origin/lsf
    def get_ext_int(self, params, camera_id):
        lidar_to_camera = params["camera%d" % camera_id]['extrinsic'].astype(np.float32) # R_cw
        camera_to_lidar = np.linalg.inv(lidar_to_camera) # R_wc
        camera_intrinsic = params["camera%d" % camera_id]['intrinsic'].astype(np.float32
        )
        return camera_to_lidar, camera_intrinsic

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask):
        """
        Given the raw point cloud, augment by flipping and rotation.
        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape
        object_bbx_center : np.ndarray
            (n, 7) shape to represent bbx's x, y, z, h, w, l, yaw
        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask