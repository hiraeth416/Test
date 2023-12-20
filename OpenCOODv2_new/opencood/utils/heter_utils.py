"""
Agent Selection Module for Heterogeneous Collaboration.

Maybe later can use data augment, one sample with different selection setting. 
"""
import numpy as np
import random
import os
from collections import OrderedDict
import json


class AgentSelector:
    def __init__(self, args, max_cav):
        self.lidar_ratio = args['lidar_ratio']
        self.ego_modality = args['ego_modality']  # 'random' / 'lidar'/ 'camera'
        self.max_cav = max_cav

        self.preset = None
        if "preset_file" in args:
            self.preset_file = args['preset_file'] # txt file
            self.preset = np.loadtxt(self.preset_file)


    def select_agent(self, i):
        """
        select agent to be equipped with LiDAR / Camera according to the strategy
        1 indicates lidar
        0 indicates camera
        """
        lidar_agent = np.random.choice(2, self.max_cav, p=[1 - self.lidar_ratio, self.lidar_ratio])

        if self.ego_modality == 'lidar':
            lidar_agent[0] = 1

        if self.ego_modality == 'camera':
            lidar_agent[0] = 0
        
        if self.preset:
            lidar_agent = self.preset[i]

        return lidar_agent, 1 - lidar_agent
    

class Adaptor:
    def __init__(self, 
                ego_modality, 
                model_modality_list, 
                modality_assignment,
                lidar_channels_dict,
                mapping_dict,
                cav_preference,
                train):
        self.ego_modality = ego_modality
        self.model_modality_list = model_modality_list
        self.modality_assignment = modality_assignment
        self.lidar_channels_dict = lidar_channels_dict
        self.mapping_dict = mapping_dict
        if cav_preference is None:
            cav_preference = dict.fromkeys(model_modality_list, 1/len(model_modality_list))
        self.cav_preferece = cav_preference # training, probability for setting non-ego cav modality
        self.train = train


    def reorder_cav_list(self, cav_list, scenario_name):
        """
        When evaluation, make the cav that could be ego modality after mapping be the first.

        This can check the training effect of aligner.

        work in basedataset -> reinitialize
        """
        if self.train:
            # shuffle the cav list
            random.shuffle(cav_list)
            return cav_list

        assignment = self.modality_assignment[scenario_name]
        if assignment[cav_list[0]] not in self.ego_modality:
            ego_cav = None
            for cav_id, modality in assignment.items():
                if self.mapping_dict[modality] in self.ego_modality: # after mapping the modality is ego
                    ego_cav = cav_id
                    break

            if ego_cav is None:
                return cav_list

            other_cav = sorted(list(assignment.keys()))
            other_cav.remove(ego_cav)
            cav_list = [ego_cav] + other_cav

        return cav_list
    
    def reassign_cav_modality(self, modality_name, idx_in_cav_list):
        """
        work in basedataset -> reinitialize
        """
        if self.train: 
            # always assign the ego_modality to idx 0 in cav_list
            if idx_in_cav_list == 0:
                return np.random.choice(self.ego_modality.split("&"))
            return random.choices(list(self.cav_preferece.keys()), weights=self.cav_preferece.values())[0]
        else:
            return self.mapping_dict[modality_name]

    def unmatched_modality(self, cav_modality):
        """
        work in 
            intermediate_heter_fusion_dataset -> __getitem__
            late_heter_fusion_dataset -> get_item_test

        Returns:
            True/False. If the input modality is in the model_modality_list
        """
        return cav_modality not in self.model_modality_list

    def switch_lidar_channels(self, cav_modality, lidar_file_path):
        """
        Currently only support OPV2V
        """
        if self.lidar_channels_dict.get(cav_modality, None) == 32:
            return lidar_file_path.replace("OPV2V","OPV2V_Hetero").replace(".pcd", "_32.pcd")
        if self.lidar_channels_dict.get(cav_modality, None) == 16:
            return lidar_file_path.replace("OPV2V","OPV2V_Hetero").replace(".pcd", "_16.pcd")
        return lidar_file_path


def assign_modality(root_dir="dataset/OPV2V", output_path="opencood/logs/heter_modality_assign/opv2v.json"):
    np.random.seed(303)
    splits = ['train', 'test', 'validate']
    scenario_cav_modality_dict = OrderedDict()

    for split in splits:
        split_path = os.path.join(root_dir, split)
        scenario_folders = sorted([os.path.join(split_path, x)
                                    for x in os.listdir(split_path) if
                                    os.path.isdir(os.path.join(split_path, x))])

        for scenario_folder in scenario_folders:
            scenario_name = scenario_folder.split('/')[-1]
            scenario_cav_modality_dict[scenario_name] = OrderedDict()

            cav_list = sorted([x for x in os.listdir(scenario_folder) \
                                if os.path.isdir(os.path.join(scenario_folder, x))])

            # randomly exclude one agent to be M3 
            M3_agent_idx = np.random.randint(len(cav_list))

            for j, cav_id in enumerate(cav_list):

                if j == M3_agent_idx:
                    scenario_cav_modality_dict[scenario_name][cav_id] = "m3" # M3 modality
                else:
                    scenario_cav_modality_dict[scenario_name][cav_id] = 'm'+str(np.random.randint(1,3)) # can be M1 or M2 mdoality
    
    with open(output_path, "w") as f:
        json.dump(scenario_cav_modality_dict, f, indent=4, sort_keys=True)


def assign_modality_4(root_dir="dataset/OPV2V", output_path="opencood/logs/heter_modality_assign/opv2v_4modality.json"):
    np.random.seed(303)
    splits = ['train', 'test', 'validate']
    # splits = ['test_64_new']
    # splits = ['test_32']
    splits = ['test_64_half']
    scenario_cav_modality_dict = OrderedDict()

    for split in splits:
        split_path = os.path.join(root_dir, split)
        scenario_folders = sorted([os.path.join(split_path, x)
                                    for x in os.listdir(split_path) if
                                    os.path.isdir(os.path.join(split_path, x))])

        for scenario_folder in scenario_folders:
            scenario_name = scenario_folder.split('/')[-1]
            scenario_cav_modality_dict[scenario_name] = OrderedDict()

            cav_list = sorted([x for x in os.listdir(scenario_folder) \
                                if os.path.isdir(os.path.join(scenario_folder, x))])

            perm = np.random.permutation(4) + 1 
            for j, cav_id in enumerate(cav_list):
                scenario_cav_modality_dict[scenario_name][cav_id] = 'm'+str(perm[j%4]) # m1 or m2 or m3 or m4

    
    with open(output_path, "w") as f:
        json.dump(scenario_cav_modality_dict, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    assign_modality_4(root_dir="/GPFS/rhome/yifanlu/workspace/dataset/OPV2V_MoreAgents/lidar/", output_path="/GPFS/public/yhu/OpenCOODv2_CodeFilling/modality_assignment_more_agent_lidar_train.json")
