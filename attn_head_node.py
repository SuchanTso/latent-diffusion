import numpy as np
from scipy.stats import entropy
class Attn_head_node:
    """
    Attn_head_node class
    """
    def __init__(self , hid ,data, ica_comp , attention_map , masks):
        self.head_id = hid
        self.ica_comp = ica_comp.T
        self.data = data
        self.attn_map = attention_map
        self.masks = masks
        self.mask_parent_list = [-1] * len(masks)

    def get_masks(self):
        return self.masks
    def get_head_id(self):
        return self.head_id

    def encode_mask_region(self , h=64, w=64):
        """
        编码mask区域的语义特征
        返回: 编码好的特征向量list [len(mask) ,m+2,]
        """
        # ICA特征: 各成分在区域内的均值
        mask_region = []
        for mask in self.masks:
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
    
    def set_mask_cluster_id(self, mask_id, cluster_id):
        self.mask_parent_list[mask_id] = cluster_id

    def get_mask_cluster_id(self, mask_id):
        return self.mask_parent_list[mask_id]
    
    def set_cluster_label(self, cluster_label):
        self.cluster_label = cluster_label
        
    def get_cluster_label(self):
        return self.cluster_label