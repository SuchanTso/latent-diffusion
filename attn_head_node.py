import numpy as np
from scipy.stats import entropy
import torch
import math
class Attn_head_node:
    """
    Attn_head_node class
    """
    def __init__(self , hid ,data, ica_comp , ica_matrix , ica_unmix_matrix , attention_map , masks):
        self.head_id = hid
        self.ica_comp = ica_comp.T
        self.ica_matrix = ica_matrix
        self.ica_unmix_matrix = ica_unmix_matrix
        self.data = data
        self.attn_map = attention_map
        self.masks = masks
        self.mask_parent_list = [-1] * len(masks)

    

    def encode_mask_region(self , h=64, w=64):
        """
        编码mask区域的语义特征
        返回: 编码好的特征向量list [len(mask) ,m+2,]
        """
        # ICA特征: 各成分在区域内的均值
        mask_region = []
        for mask in self.masks:
            h = w = int(math.sqrt(mask.shape[0]))
            ica_means = np.mean(self.ica_comp[:, mask], axis=1)  # [m,]
            
            # 注意力特征: 区域内注意力的熵（衡量专注度）
            eps = 1e-3
            attn_map_mean = np.mean(np.clip(self.attn_map[mask].cpu().numpy() , eps , None), axis=0)
            attn_entropy = entropy(attn_map_mean)
            
            # 空间特征: 区域中心坐标
            positions = np.argwhere(mask.reshape(h, w))
            if positions.size == 0:
                center_x, center_y = h / 2, w / 2  # 处理空掩码情况
            else:
                center_y, center_x = positions.mean(axis=0)
            # print(f"ica_means:{ica_means} , attn_entropy:{attn_entropy} , center_x:{center_x} , center_y:{center_y}")
            mask_region.append(np.concatenate([ica_means, [attn_entropy, center_x, center_y]]))
        return mask_region
    
    def compute_mask_region_energy(self):
        """
        计算mask区域的能量
        返回: { (head_id, mask_id): energy
        """
        energy_dict = {}
        for mid, mask in enumerate(self.masks):
            # 能量 = ICA成分方差均值 + 注意力均值
            ica_var = np.mean(np.var(self.ica_comp[:, mask], axis=1))
            attn_mean = np.mean(self.attn_map[mask].cpu().numpy())
            energy_dict[(self.head_id, mid)] = 0.7*ica_var + 0.3*attn_mean
        return energy_dict

    def find_independent_comp_by_mask_id(self, mask_id):
        """
        mask_id:使用统一后的id而不是uuid
        根据mask_id找到对应的独立成分
        note: find the max correlated one for now
        return: -1 if not found
        """
        correlated_index = -1
        if mask_id in self.cluster_label:
            mask = self.cluster_label == mask_id
            print(f"cluster_label.min = {self.cluster_label.min()} ,cluster_label.max = {self.cluster_label.max()},mask.shape:{mask.shape}")
            mean_cluster = np.mean(self.ica_comp.T[mask,:], axis=0)
            global_mean = np.mean(self.ica_comp.T, axis=0)
            delta = np.abs(mean_cluster - global_mean)
            # print(f"mask_id:{mask_id} , mean_cluster:{mean_cluster}\nglobal_mean:{global_mean} , delta:{delta}")
            key_comp = np.argmax(delta)
            print(f"key_comp:{key_comp}")
            correlated_index = key_comp
        else:
            print(f"mask_id:{mask_id} not in head_{self.head_id} cluster_label")
        return correlated_index
    
    def gen_independent_comp_by_data(self , data):
        """
        根据data生成独立成分
        """
        # S =  data @ ica_unmix_matrix
        return data @ self.ica_unmix_matrix
    
    def reconstruct_data(self , independent_comp):
        """
        重构数据
        """
        print(f"head_{self.head_id} independent_comp.shape:{independent_comp.shape} , ica_matrix.shape:{self.ica_matrix.shape}")
        recons_data = independent_comp @ self.ica_matrix.T
        print(f"diff:{np.abs(recons_data - self.data.cpu().numpy()).mean()}")
        return independent_comp @ self.ica_matrix.T
    
    def compare_indenpent_comp(self , new_comp , cluster_index):
        """
        比较独立成分
        """
        # 计算相关性
        origin_comp = self.ica_comp.T
        assert origin_comp.shape == new_comp.shape
        final_comp = np.zeros_like(origin_comp)
        for i in range(origin_comp.shape[0]):
            final_comp[i] = origin_comp[i] if i == cluster_index else new_comp[i]
        return final_comp
    
#=======================================get_set===================================================
    def get_masks(self):
        return self.masks
    
    def get_head_id(self):
        return self.head_id

    def set_mask_cluster_id(self, mask_id, cluster_id):
        self.mask_parent_list[mask_id] = cluster_id

    def get_mask_cluster_id(self, mask_id):
        return self.mask_parent_list[mask_id]
    
    def set_cluster_label(self, cluster_label):
        self.cluster_label = cluster_label
        
    def get_cluster_label(self):
        return self.cluster_label
    
    def get_ica_matrix(self):
        return self.ica_matrix
    
    def get_ica_unmix_matrix(self):
        return self.ica_unmix_matrix

    def get_origin_data(self):
        return self.data
    
    def get_ica_comp(self):
        return self.ica_comp