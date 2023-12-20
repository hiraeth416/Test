# -*- coding: utf-8 -*-
# Author: Yue Hu <phyllis1sjtu@outlook.com>
# License: TDG-Attribution-NonCommercial-NoDistrib

import torch
import torch.nn as nn
import numpy as np
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple

class Communication(nn.Module):
    def __init__(self, args):
        super(Communication, self).__init__()
        
        self.smooth = False
        self.thre = args['thre']
<<<<<<< HEAD
        # self.R_thre = args['R_thre']
=======
        if 'solver' in args:
            self.solver = True
            self.solver_thre = args['solver']['thre']
            self.solver_method = args['solver']['method']
        else:
            self.solver = False
>>>>>>> origin/lsf
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
<<<<<<< HEAD
        self.debug = False #True # 
=======
>>>>>>> origin/lsf
        self.vis_count = 0
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

<<<<<<< HEAD
    def forward_single(self, batch_confidence_maps, record_len, pairwise_t_matrix):
=======
    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix):
>>>>>>> origin/lsf
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
<<<<<<< HEAD
            # t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) # dim1=2 represents the confidence of two anchors
=======
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            ori_communication_maps = batch_confidence_maps[b].sigmoid().max(dim=1)[0].unsqueeze(1) # dim1=2 represents the confidence of two anchors [n,1,h,w]
            #print("-------------------comm------------------------")
            #print(ori_communication_maps.shape)
>>>>>>> origin/lsf
            
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps

            ones_mask = torch.ones_like(communication_maps).to(communication_maps.device)
            zeros_mask = torch.zeros_like(communication_maps).to(communication_maps.device)
<<<<<<< HEAD
            communication_mask = torch.where(communication_maps>self.thre, ones_mask, zeros_mask)

            communication_rate = communication_mask[0].sum()/(H*W)

            # communication_mask = warp_affine_simple(communication_mask,
            #                                 t_matrix[0, :, :, :],
            #                                 (H, W))
            
            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
            communication_mask_nodiag[::2] = ones_mask[::2]
=======
            communication_mask=ones_mask.clone()

            if self.solver:
                #print(" in solver ") 
                # import ipdb; ipdb.set_trace()
                ########### warp to ego coordinate ###########
                #print("communication_maps",communication_maps.shape)
                #print("t_matrix", t_matrix.shape)
                ego_communication_maps = warp_affine_simple(communication_maps,
                                                t_matrix[0, :, :, :],
                                                (H, W))
                ego_communication_mask = warp_affine_simple(communication_mask,
                                                t_matrix[0, :, :, :],
                                                (H, W))
                
                if self.solver_method == 'sum':
                    
                    N,_,H,W = ego_communication_maps.shape
                    ego_communication_maps = ego_communication_maps.view(N,-1)

                    sorted_communications, indices = ego_communication_maps[1:].sort(dim=0, descending=True)
                    cum_sum = torch.cat([ego_communication_maps[0:1],sorted_communications],dim=0).cumsum(dim=0)
                    
                    below_thre = cum_sum < self.solver_thre

                    _, sorted_indices = indices.sort(dim=0)
                    ego_communication_mask = torch.gather(below_thre[1:], 0, sorted_indices)
                    ego_communication_mask = torch.cat([below_thre[0:1], ego_communication_mask], dim=0)

                    ego_communication_mask = ego_communication_mask.view(N,1,H,W) * 1.0
                    ego_communication_maps = ego_communication_maps.view(N,1,H,W)

                    # # import ipdb; ipdb.set_trace()
                    # vis_communication_maps = ego_communication_maps
                    # # vis_communication_mask = ego_communication_mask
                    # vis_communication_mask = -(ego_communication_maps * torch.log2(ego_communication_maps+1e-6) + (1-ego_communication_maps)*torch.log2(1-ego_communication_maps+1e-6))
                    # self.vis_CR_mask(vis_communication_maps, vis_communication_mask)
                    # self.vis_count += 1
                    
                elif self.solver_method == 'max':
                    max_maps = ego_communication_maps.max(dim=0, keepdim=True)[0]
                    # print("max_maps shape: ", max_maps.shape)
                    ego_communication_mask = torch.where(ego_communication_maps>=max_maps, ones_mask, zeros_mask)
                
                ########### warp to ego coordinate ###########
                communication_mask = warp_affine_simple(ego_communication_mask,
                                                t_matrix[:, 0, :, :],
                                                (H, W))
  
            # import ipdb; ipdb.set_trace()
            communication_mask_by_conf = torch.where(communication_maps>self.thre, ones_mask, zeros_mask)
            communication_mask = communication_mask_by_conf * communication_mask

            diff = (communication_mask_by_conf - communication_mask).abs().sum() / (communication_mask_by_conf.sum()+1e-6)
            #print('diff_ratio: {:.6f}'.format(diff.item()))

            # import ipdb; ipdb.set_trace()

            # communication_rate = communication_mask[0].sum()/(H*W)
            if N > 1:
                communication_rate = communication_mask[1:].sum()/(H*W*(N-1))
            else:
                communication_rate = torch.tensor(0.0).to(communication_mask.device)

            communication_mask_nodiag = communication_mask.clone()
            ones_mask = torch.ones_like(communication_mask).to(communication_mask.device)
            communication_mask_nodiag[0] = ones_mask[0]
>>>>>>> origin/lsf

            communication_masks.append(communication_mask_nodiag)
            communication_rates.append(communication_rate)
            batch_communication_maps.append(ori_communication_maps*communication_mask_nodiag)
        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks, communication_rates
    
<<<<<<< HEAD
    def forward(self, batch_confidence_maps, record_len, pairwise_t_matrix, batch_request_maps=None, config_thre=True):
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape

        # if not config_thre:
        #     self.thre = 0.0

        # comm_thre_list = [0, 0.001, 0.01, 0.03, 0.10, 0.30, 0.60, 1.0]
        # if config_thre:
        #     for i, thre in enumerate(comm_thre_list):
        #         if thre > self.thre:
        #             break
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            curr_comm_maps = []
            comm_mask_list = []
            for i in range(N):
                if batch_request_maps is not None:
                    curr_request_maps = batch_request_maps[b][i:i+1]
                    # curr_request_mask = 1.0 - ((1 - curr_request_maps) > self.R_thre) * 1.0
                    # curr_request_maps = curr_request_mask
                    request_maps = warp_affine_simple(curr_request_maps.expand(N,-1,-1,-1),
                                            t_matrix[:, i, :, :],
                                            (H, W)) # warp request map of agent i to other agents' coordnate
                    comm_maps = batch_confidence_maps[b] * request_maps # (N, 1, H, W)
                    if self.debug:
                        self.vis_CR_mask('vis_cr_masks', batch_confidence_maps[b], request_maps)
                        self.vis_count += 1
                else:
                    comm_maps = batch_confidence_maps[b] # (N, 1, H, W)
                curr_comm_maps.append(comm_maps.unsqueeze(0))
            curr_comm_maps = torch.cat(curr_comm_maps, dim=0) # (N, N, 1, H, W)
            curr_comm_maps = curr_comm_maps.flatten(0,1)

            if self.smooth:
                curr_comm_maps = self.gaussian_filter(curr_comm_maps)

            # if config_thre:
            #     comm_mask = (curr_comm_maps>thre) * 1.0
            # else:
            comm_mask = (curr_comm_maps>self.thre) * 1.0

            comm_rate = comm_mask.sum()/(H*W*N*N)
            
            comm_mask_nodiag = comm_mask.clone()
            ones_mask = torch.ones_like(comm_mask).to(comm_mask.device)
            comm_mask_nodiag[::N+1] = ones_mask[::N+1] 
            # comm_mask_nodiag = ones_mask

            communication_masks.append(comm_mask_nodiag)
            communication_rates.append(comm_rate)
            batch_communication_maps.append(curr_comm_maps*comm_mask_nodiag)
        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks, communication_rates
    
    def forward_CR(self, batch_confidence_maps, record_len, pairwise_t_matrix, batch_request_maps=None):
        # batch_confidence_maps:[(L1, H, W), (L2, H, W), ...]
        # pairwise_t_matrix: (B,L,L,2,3)
        # thre: threshold of objectiveness
        # a_ji = (1 - q_i)*q_ji
        B, L, _, _, _ = pairwise_t_matrix.shape
        _, _, H, W = batch_confidence_maps[0].shape
        
        communication_masks = []
        communication_rates = []
        batch_communication_maps = []
        for b in range(B):
            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            curr_comm_maps = []
            for i in range(N):
                if batch_request_maps is not None:
                    request_maps = warp_affine_simple(batch_request_maps[b][i:i+1].expand(N,-1,-1,-1),
                                            t_matrix[:, i, :, :],
                                            (H, W)) # warp request map of agent i to other agents' coordnate
                    comm_maps = batch_confidence_maps[b] * request_maps # (N, 1, H, W)
                    if self.debug:
                        self.vis_CR_mask('vis_cr_masks', batch_confidence_maps[b], request_maps)
                        self.vis_count += 1
                else:
                    comm_maps = batch_confidence_maps[b] # (N, 1, H, W)
                curr_comm_maps.append(comm_maps.unsqueeze(0))
            curr_comm_maps = torch.cat(curr_comm_maps, dim=0) # (N, N, 1, H, W)
            curr_comm_maps = curr_comm_maps.flatten(0,1)

            if self.smooth:
                curr_comm_maps = self.gaussian_filter(curr_comm_maps)

            comm_mask = (curr_comm_maps>self.thre) * 1.0

            comm_rate = comm_mask.sum()/(H*W*N*N)
            
            comm_mask_nodiag = comm_mask.clone()
            ones_mask = torch.ones_like(comm_mask).to(comm_mask.device)
            comm_mask_nodiag[::N+1] = ones_mask[::N+1] 
            # comm_mask_nodiag = ones_mask

            communication_masks.append(comm_mask_nodiag)
            communication_rates.append(comm_rate)
            batch_communication_maps.append(curr_comm_maps*comm_mask_nodiag)
        communication_rates = sum(communication_rates)/B
        communication_masks = torch.concat(communication_masks, dim=0)
        return batch_communication_maps, communication_masks, communication_rates

    def vis_CR_mask(self, save_name, confidence_maps_list, request_maps_list):
        import seaborn as sns
        import matplotlib.pyplot as plt
        save_dir = '/remote-home/share/yhu/Co_Flow/opencood/visualization/{}/{}_{:.04f}.png'.format(save_name, self.vis_count, self.thre)
        fig, axes = plt.subplots(len(confidence_maps_list), 3, figsize=(20, 9))
        # fig, axes = plt.subplots(len(confidence_maps_list), 5, figsize=(28, 9))
        for ax_i, _ in enumerate(confidence_maps_list):
            conf_single = (confidence_maps_list[ax_i].cpu().numpy()[0]>self.thre)*1.0
            # conf_single = confidence_maps_list[ax_i].cpu().numpy()[0]
            total_size = conf_single.shape[0] * conf_single.shape[1]
            rate_single = conf_single.sum()/total_size
=======

    def vis_CR_mask(self, confidence_maps, communication_masks):
        import seaborn as sns
        import matplotlib.pyplot as plt
        save_dir = '/GPFS/public/yhu/OpenCOODv2_CodeFilling/opencood/visualization/vis_confmaps/{}_{:.04f}.png'.format(self.vis_count, self.thre)
        fig, axes = plt.subplots(len(confidence_maps), 3, figsize=(20, 9))
        # fig, axes = plt.subplots(len(confidence_maps_list), 5, figsize=(28, 9))
        for ax_i, _ in enumerate(confidence_maps):
            # conf_single = (confidence_maps[ax_i].cpu().numpy()[0]>self.thre)*1.0
            conf_single = confidence_maps[ax_i].cpu().numpy()[0]
            total_size = conf_single.shape[0] * conf_single.shape[1]
            rate_single = (conf_single>self.thre).sum()/total_size
>>>>>>> origin/lsf
            sns.heatmap(conf_single, ax=axes[ax_i][0], cbar=True)
            axes[ax_i][0].set_xticks([])
            axes[ax_i][0].set_yticks([])
            axes[ax_i][0].set_title('t_{}_conf:{:.06f}'.format(ax_i, rate_single))

<<<<<<< HEAD
            # conf_whist = request_maps_list[ax_i].cpu().numpy()[0]
            conf_whist = (confidence_maps_list[ax_i].cpu().numpy()[0] * request_maps_list[ax_i].cpu().numpy()[0]>self.thre)*1.0
            rate_whist = conf_whist.sum()/total_size
=======
            conf_whist = communication_masks[ax_i].cpu().numpy()[0]
            # conf_whist = (communication_masks[ax_i].cpu().numpy()[0] * request_maps_list[ax_i].cpu().numpy()[0]>self.thre)*1.0
            rate_whist = ((conf_single>self.thre) * conf_whist).sum()/total_size
>>>>>>> origin/lsf
            sns.heatmap(conf_whist, ax=axes[ax_i][1], cbar=True)
            axes[ax_i][1].set_xticks([])
            axes[ax_i][1].set_yticks([])
            axes[ax_i][1].set_title('t_{}_whist:{:.06f}'.format(ax_i, rate_whist))

<<<<<<< HEAD
            conf_diff = conf_single - conf_whist
=======
            conf_diff = (conf_single>self.thre)*1.0 - (conf_single>self.thre)*1.0 * conf_whist
>>>>>>> origin/lsf
            rate_diff = conf_diff.sum()/total_size
            sns.heatmap(conf_diff, ax=axes[ax_i][2], cbar=True)
            axes[ax_i][2].set_xticks([])
            axes[ax_i][2].set_yticks([])
            axes[ax_i][2].set_title('t_{}_diff:{:.06f}'.format(ax_i, rate_diff))
        fig.tight_layout()
        plt.savefig(save_dir)
        plt.close()