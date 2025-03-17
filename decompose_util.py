import os
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import transforms as T
from math import sqrt
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image
import torch
import torch.nn.functional as F
import itertools
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from pnp_utils import decompose_spectral , decompose_q_high_var_ica
from attn_head_node import Attn_head_node
from DSU import mark_roots

head_dict = {}

def compute_spatial_overlap(mask_a, mask_b):
    """
    计算两个成分的空间重叠度(IoU)
    参数:
        mask_a: comp_a对应的聚类掩码 [seq_len,] bool
        mask_b: comp_b对应的聚类掩码 [seq_len,] bool
    """
    # 计算掩码交集与并集
    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    return intersection / union if union > 0 else 0.0


def validate_independence(comp_a, comp_b, bins=20):
    """
    通过联合分布熵验证独立性
    返回值: 0-1之间,越接近1表示越独立
    """
    # 计算联合直方图
    hist_2d, x_edges, y_edges = np.histogram2d(comp_a, comp_b, bins=bins)
    joint_prob = hist_2d / hist_2d.sum()
    
    # 计算边际分布
    prob_a = np.sum(joint_prob, axis=1)
    prob_b = np.sum(joint_prob, axis=0)
    
    # 计算互信息
    mi = np.sum(joint_prob * np.log(joint_prob / (prob_a[:,None] * prob_b[None,:] + 1e-8)))
    # 标准化到0-1
    return 1.0 - mi / (entropy(prob_a) + entropy(prob_b))

def gram_schmidt_ortho(components):
    """ Gram-Schmidt正交化 """
    basis = []
    for vec in components:
        v = vec.copy()
        for b in basis:
            v -= np.dot(v, b) * b
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            basis.append(v / norm)
    return np.column_stack(basis)

# def auto_ortho_merge(all_components, cluster_masks, 
#                     iou_threshold=0.5, indep_threshold=0.7):
#     """
#     all_components: 所有成分的字典 {head_id: [comp1, comp2, ...]}
#     cluster_masks: 所有成分的聚类掩码 {head_id: [mask1, mask2, ...]}
#     无监督正交合并主函数
#     返回: 正交化后的成分列表
#     """
#     # 收集所有成分及其元数据
#     component_pool = []
#     for hid in all_components:
#         for cid, comp in enumerate(all_components[hid]):
#             component_pool.append({
#                 "head": hid,
#                 "cid": cid,
#                 "comp": comp,
#                 "mask": cluster_masks[hid] # len(mask) == 3 , list
#             })
    
#     # 两两检测冗余
#     to_merge = []
#     for i, j in itertools.combinations(range(len(component_pool)), 2):
#         comp_i = component_pool[i]
#         comp_j = component_pool[j]
        
#         iou = compute_spatial_overlap(
#             comp_i["mask"], comp_j["mask"]
#         )
#         indep_score = validate_independence(comp_i["comp"], comp_j["comp"])
        
#         if iou > iou_threshold and indep_score < indep_threshold:
#             to_merge.append( (i,j) )
    
#     # 合并冗余成分（保留方差最大的）
#     kept_indices = set(range(len(component_pool)))
#     for i,j in to_merge:
#         var_i = np.var(component_pool[i]["comp"])
#         var_j = np.var(component_pool[j]["comp"])
#         remove_idx = i if var_i < var_j else j
#         if remove_idx in kept_indices:
#             kept_indices.remove(remove_idx)
    
#     # 提取保留成分并正交化
#     kept_comps = [component_pool[i]["comp"] for i in kept_indices]
#     return kept_comps#gram_schmidt_ortho(kept_comps)


def encode_head_data(hid , q , ica_comps , attention_map , masks):
    head = Attn_head_node(hid , q , ica_comps , attention_map , masks)
    head_dict[hid] = head
    return head
    # return {
    #     "hid": hid,
    #     "ica_comps": ica_comps.T,
    #     "attn_map": attention_map,
    #     "masks": masks
    # }

# def encode_mask_region(head_data, mask , h=64, w=64):
#     """
#     编码单个mask区域的语义特征
#     返回: 特征向量 [m+2,]
#     """
#     # ICA特征: 各成分在区域内的均值
#     ica_means = np.mean(head_data['ica_comps'][:, mask], axis=1)  # [m,]
    
#     # 注意力特征: 区域内注意力的熵（衡量专注度）
#     eps = 1e-3
#     attn_map_mean = np.mean(np.clip(head_data['attn_map'][mask].cpu().numpy() , eps , None), axis=0)
#     attn_entropy = entropy(attn_map_mean)
    
#     # 空间特征: 区域中心坐标
#     positions = np.argwhere(mask.reshape(h, w))
#     if positions.size == 0:
#         center_x, center_y = h / 2, w / 2  # 处理空掩码情况
#     else:
#         center_y, center_x = positions.mean(axis=0)
#     # print(f"ica_means:{ica_means} , attn_entropy:{attn_entropy} , center_x:{center_x} , center_y:{center_y}")
#     return np.concatenate([ica_means, [attn_entropy, center_x, center_y]])


def detect_redundant_masks(head1, head2, sim_threshold=0.9, iou_threshold=0.4):
    """
    检测两Head间的冗余mask对
    返回: 冗余对列表 [(mask1_idx, mask2_idx, similarity)]
    """
    redundant_pairs = []
    
    # 编码所有mask特征
    feat1 = head1.encode_mask_region()
    feat2 = head2.encode_mask_region()
    # feat1 = [encode_mask_region(head1, m) for m in head1['masks']]
    # feat2 = [encode_mask_region(head2, m) for m in head2['masks']]
    
    # 计算两两相似度
    mask_list_1 = head1.get_masks()
    mask_list_2 = head2.get_masks()
    for i, f1 in enumerate(feat1):
        for j, f2 in enumerate(feat2):
            # 特征余弦相似度
            feat_sim = 1 - cosine(f1, f2)
            # 空间IoU
            iou = compute_spatial_overlap(mask_list_1[i], mask_list_2[j])
            # iou = np.sum(mask_list_1[i] & mask_list_2[j]) / np.sum(mask_list_1[i] | mask_list_2[j])
            # 综合相似度
            combined_sim = 0.6*feat_sim + 0.4*iou
            redundant = False#debug info
            if combined_sim > sim_threshold:
                redundant = True
                redundant_pairs.append(((head1.get_head_id(), i), (head2.get_head_id(), j), combined_sim))    
            print(f"mask_{(head1.get_head_id(), i)} and mask_{(head2.get_head_id(), j)}:feat_sim:{feat_sim:.2f} , iou:{iou:.2f} , combined_sim:{combined_sim:.2f} , redundant:{redundant}")
    return redundant_pairs

def gen_mask_uuid():
    import time
    """
    使用当前时间的微秒数生成一个8位（1字节）的唯一标识符。
    
    返回:
        uint8_id: 一个介于0到255之间的整数，基于当前时间的微秒数。
    """
    return int(time.time() * 1e6) % 256

def gen_mask_node_tree(all_heads , mask_parent):
    """
    生成mask的树结构,将mask根据剪枝结果合并
    """
    for hid , head in enumerate(all_heads):
        head_masks = head.get_masks()
        for mid , mask in enumerate(head_masks):
            if (hid, mid) in mask_parent:
                # already grouped this mask to another one , find the root
                mask_cluster_id = find_mask_root(mask_parent,all_heads , (hid, mid))
            else:
                mask_cluster_id = gen_mask_uuid()
            print(f"set mask_{hid}_{mid}:{mask_cluster_id}")
            head.set_mask_cluster_id(mid , mask_cluster_id)
    pass

def gen_mask_node_tree(root_map , all_heads):
    """
    生成mask的树结构,将mask根据剪枝结果合并
    """
    for hid , head in enumerate(all_heads):
        head_masks = head.get_masks()
        for mid , mask in enumerate(head_masks):
            key = (hid, mid)
            root = root_map[key] if key in root_map else key
            if key == root:
                if head.get_mask_cluster_id(mid) == -1:
                    mask_cluster_id = gen_mask_uuid()
                    head.set_mask_cluster_id(mid , mask_cluster_id)
            else:
                root_hid , root_mid = root
                root_head = head_dict[root_hid]
                root_mask_cluster_id = root_head.get_mask_cluster_id(root_mid)
                if root_mask_cluster_id == -1:
                    cluster_id = gen_mask_uuid()
                    root_head.set_mask_cluster_id(root_mid , cluster_id)
                    head.set_mask_cluster_id(mid , cluster_id)
                else:
                    cluster_id = root_mask_cluster_id
                    head.set_mask_cluster_id(mid , root_mask_cluster_id)





def find_mask_root(mask_parent,all_heads ,  mask_id):
    """
    递归查找mask的根节点
    """
    hid , mid = mask_id
    if mask_id in mask_parent:
        return find_mask_root(mask_parent,all_heads , mask_parent[mask_id])
    else:
        head = None
        for h in all_heads:
            if h.get_head_id() == hid:
                head = h
                break
        assert head is not None
        mask_cluster_id = head.get_mask_cluster_id(mid)
        if mask_cluster_id == -1:
            mask_cluster_id = gen_mask_uuid()
            head.set_mask_cluster_id(mid , mask_cluster_id)
        return mask_cluster_id

def sort_redundant_pairs(key1 , key2):
    """
    对冗余对进行排序
    """
    key1_f , key1_s = key1
    key2_f , key2_s = key2
    
    if key1_f < key2_f:
        return key1 , key2
    elif key1_f > key2_f:
        return key2 , key1
    else:
        if key1_s < key2_s:
            return key1 , key2
        else:
            return key2 , key1

def set_parent_mask(mask_parent , mask_id_1 , mask_id_2):
    """
    给mask_id_1 , mask_id_2设置父子关系
    mask_id_small考虑作为父节点
    如果mask_id_big已经有父节点,则将mask_id_bing的父结点设置为mask_id_small的root结点的父结点
    """
    mask_id_small , mask_id_big = sort_redundant_pairs(mask_id_1 , mask_id_2)
    if mask_id_big not in mask_parent:
        small_parent = mask_id_small
        while small_parent in mask_parent:
            if small_parent == mask_id_big:
                return
            small_parent = mask_parent[small_parent]
        mask_parent[mask_id_big] = mask_id_small
    else:
        set_parent_mask(mask_parent , mask_parent[mask_id_big] , mask_id_small)
    

def prune_redundant_masks(all_heads, redundant_pairs):
    """
    输入: 
        all_heads: 所有Head的数据列表 [head1, head2, ...]
        redundant_pairs: detect_redundant_masks的输出
    """
    #TODO: make the right pruning decision @Suchan
    # 计算每个mask的能量
    energy_dict = {}
    mask_parent = {} # record the belonging relationship between masks
    for  head in all_heads:
        energy_dict.update(head.compute_mask_region_energy())
        
    
    # 处理冗余对
    redundant_dict = {}
    kept_masks = set(energy_dict.keys())
    for (hid1, mid1), (hid2, mid2), sim in redundant_pairs:
        key1 = (hid1, mid1)
        key2 = (hid2, mid2)
        
        if key1 in kept_masks and key2 in kept_masks:
            if energy_dict[key1] > energy_dict[key2]:
                kept_masks.remove(key2)
                # mask_parent[key2] = key1
            else:
                kept_masks.remove(key1)
                # mask_parent[key1] = key2
        redundant_dict[key1] = key2
            # redundant_dict[key2] = key1
            
        

    root_map = mark_roots(redundant_dict)
    
    gen_mask_node_tree(root_map , all_heads)
    #TODO: mask merge is not right , need to fix it @Suchan

    #===============debug function================
    # for hid , head in enumerate(all_heads):
    #     mask_list = head.get_masks()
    #     for mid , mask in enumerate(mask_list):
    #         cluster_id = head.get_mask_cluster_id(mid)
    #         assert cluster_id != -1
    #         print(f"{hid}_{mid}:mask_cluster_id = {cluster_id}")
    #================debug function===============


    # 收集保留的mask数据
    #TODO: translate mask uuid to universal mask index for visualization@Suchan
    pruned_heads = []
    # for hid, head in enumerate(all_heads):
    #     kept_indices = [mid for (h, mid) in kept_masks if h == hid]
    #     pruned_head = {
    #         'hid': head['hid'],
    #         'ica_comps': head['ica_comps'],
    #         'attn_map': head['attn_map'],
    #         'masks': [head['masks'][i] for i in kept_indices],
    #     }
    #     pruned_heads.append(pruned_head)
    mask_index_map = {}
    for head in all_heads:
        hid = head.get_head_id()
        mask_list = head.get_masks()
        cluster_labels = np.zeros((4096,), dtype=int)
        for mid , mask in enumerate(mask_list):
            cluster_id = head.get_mask_cluster_id(mid)
            mask_index = translate_mask_uuid_2_index(mask_index_map , cluster_id)
            cluster_labels[mask] = mask_index
            print(f"hid:{hid} , mid:{mid}: {cluster_id}->{mask_index}")
        head.set_cluster_label(cluster_labels)
    return all_heads

def translate_mask_uuid_2_index(mask_index_map , mask_uuid):
    """
    将mask的uuid转换为全局的mask索引
    """
    if mask_uuid in mask_index_map:
        return mask_index_map[mask_uuid]
    else:
        index = len(mask_index_map) + 1 # skip 0
        mask_index_map[mask_uuid] = index
        return index


def cluster_high_var_q(q , num_components_from_high_var_q ,num_decompose_components, att_matrix_selected):
    """
    对高方差的Q向量进行聚类
    返回: 编码后的head_data , 初步的聚类标签
    """
    # 提取高方差位置
    head_data_list = []
    cluster_label_list = []
    for i , high_var_q in enumerate(q):
        S , A = decompose_q_high_var_ica(high_var_q , n_components = num_components_from_high_var_q)
        # S: [4096 , 5]
        combined_features = np.concatenate([S, att_matrix_selected[i].cpu()], axis=1)
        cluster_label = decompose_spectral(combined_features , n_components = num_decompose_components)
        # 生成区域掩码
        # print(f"cluster_label.shape = {cluster_label.shape}")
        area_masks = [cluster_label == i for i in range(num_decompose_components)]
        print(f"len(area_masks) = {len(area_masks)} ,  area_masks[0].shape = {area_masks[0].shape}")
        cluster_label_list.append(cluster_label)
        # print(f"cluster.max = {cluster_label.max()} , cluster.min = {cluster_label.min()}")
        head_data_list.append(encode_head_data(i ,high_var_q , S , att_matrix_selected[i] , area_masks))
    # pruned_heads = merge_overlap_heads(head_data_list)
    return head_data_list , cluster_label_list

def translate_head_label(head_data):
    """
    将mask列表转换为标签列表
    """
    if len(head_data['masks']) == 0:
        return np.zeros((4096,), dtype=int)
    
    mask_shape = head_data['masks'][0].shape
    labels = np.zeros(mask_shape, dtype=int)
    for idx, bool_array in enumerate(head_data['masks']):
        labels[bool_array] = idx  # 根据布尔值赋值
    return labels

def translate_head_label_new(head_data):
    """
    将mask列表转换为标签列表
    """
    return head_data.get_cluster_label()

def classify_q_by_mask(q , masks , graph_size = (64,64)):
    """""
    根据掩码对Q进行分类
    参数:
        q: 单独一个head对应的Q向量 [64, 64, dim]
        mask: 掩码 [num_head , 4096 , ]
    """""
    from collections import defaultdict
    # 初始化：每个头维护一个Cluster到Q向量的映射
    cluster_blocks = defaultdict(dict)  # 格式: {head_idx: {cluster_id: q_tensor}}

    num_heads = len(masks)
    for head_idx in range(num_heads):
        # 提取当前头的Q和Mask
        q_head = q[head_idx]  # [64, 64, dim]
        mask_head = masks[head_idx].reshape(graph_size)  # [64, 64]
        
        # 按Cluster收集Q向量
        clusters = torch.unique(mask_head)
        head_clusters = {}
        for cluster_id in clusters:
            # 获取属于该Cluster的所有位置
            positions = (mask_head == cluster_id).nonzero(as_tuple=False)  # [num_points, 2]
            # 提取对应的Q向量
            q_vectors = q_head[positions[:, 0], positions[:, 1]]  # [num_points, dim]
            head_clusters[cluster_id.item()] = q_vectors
        cluster_blocks[head_idx] = head_clusters
    return cluster_blocks

def replace_cluster_q(q_original, head_idx, cluster_id, new_q, cluster_blocks):
    """
    替换指定头和Cluster的Q向量
    - q_original: 原始Q张量 [num_heads, H, W, dim]
    - head_idx: 目标头索引
    - cluster_id: 目标Cluster ID
    - new_q: 新的Q向量 [num_points, dim]
    - cluster_blocks: 分块映射字典
    """
    q_edited = q_original.clone()
    # 获取原始Cluster的位置信息
    original_positions = cluster_blocks[head_idx][cluster_id]["positions"]
    # 确保新Q向量数量与位置匹配
    assert len(original_positions) == new_q.shape[0]
    # 替换Q值
    for i, (x, y) in enumerate(original_positions):
        q_edited[head_idx, x, y] = new_q[i]
    return q_edited