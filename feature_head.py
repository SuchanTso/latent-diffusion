import argparse, os
from tqdm import trange
import torch
from einops import rearrange
from omegaconf import OmegaConf
import json
from feature_extraction import load_model
import numpy as np
from tqdm import tqdm
import math

from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt


from decompose_util import detect_redundant_masks , prune_redundant_masks , translate_head_label_new , cluster_high_var_q


def load_experiments_features(feature_maps_paths, block, feature_type, t):
    feature_maps = []
    feature_map_origin_list = []
    for i, feature_maps_path in enumerate(feature_maps_paths):
        if "attn" in feature_type: 
            att_path = os.path.join(feature_maps_path, f"{block}_{feature_type}_time_{t}.pt")
            feature_map_origin = torch.load(att_path)
            feature_map_no_rearrange = feature_map_origin
            feature_map = rearrange(feature_map_no_rearrange, 'h n d -> n (h d)')
            # print(f"feature_type:{feature_type} , feature_origin.shape = {feature_map_origin.shape} , feature_map_no_rearrange.shape = {feature_map_no_rearrange.shape}\
            #     #    feature_map.shape = {feature_map.shape}")
            feature_map_origin_list.append(feature_map_origin)
        else:
            feature_map = \
                torch.load(os.path.join(feature_maps_path, f"{block}_{feature_type}_time_{t}.pt"))[0]#1 for origin
            feature_map_origin_list.append(feature_map)
            feature_map = feature_map.reshape(feature_map.shape[0], -1).t()  # N X C
        feature_maps.append(feature_map)


    return feature_maps , feature_map_origin_list

def load_experiments_heads(feature_maps_paths, block, feature_type, t):
    head_maps = []
    for i, feature_maps_path in enumerate(feature_maps_paths):
        if "attn" in feature_type: 
            att_head_path = os.path.join(feature_maps_path, f"{block}_self_attn_head_time_{t}.pt")
            feature_head_origin = torch.load(att_head_path)
        head_maps.append(feature_head_origin)
    return head_maps

def load_experiments_att_matrix(feature_maps_paths, block, feature_type, t):
    att_matrix_maps = []
    for i, feature_maps_path in enumerate(feature_maps_paths):
        if "attn" in feature_type: 
            att_head_path = os.path.join(feature_maps_path, f"{block}_self_attn_per_head_time_{t}.pt")
            feature_head_origin = torch.load(att_head_path)
        att_matrix_maps.append(feature_head_origin)
    return att_matrix_maps

def spatial_variance_filter(q):
    # q:[heads, seq_len, dim_head]
    variances = torch.var(q, dim=1).mean(dim=1)  # [heads]
    mask = variances > torch.median(variances)
    return q[mask], mask.nonzero().flatten().tolist()

def process_masks(all_head):
    # 获取序列长度
    seq_len = len(translate_head_label_new(all_head[0]))
    
    # 初始化结果列表
    result = []
    
    # 遍历 i 从 1 到 10
    for i in range(1, 11):
        # 创建一个全零数组，形状为 (seq_len,)
        mask_i = np.zeros(seq_len, dtype=int)
        
        # 遍历每个 mask，检查是否在该位置等于 i
        for head in all_head:
            mask = translate_head_label_new(head)
            # 如果当前位置等于 i，则设置为 1
            mask_i = np.where(mask == i, 1, mask_i)
        
        # 将生成的 mask 添加到结果中
        result.append(mask_i)
    
    return result


def visualise_cluster_masks(clusters , pruned_heads , save_dir , t , grid_size = (64 , 64) , fig_size = (12 , 6)):

    # 将一维标签转换为二维网格
    assert len(clusters) == 8 # make 4 graphs at once
    h, w = grid_size

    from matplotlib.colors import ListedColormap,LinearSegmentedColormap
    # 自定义颜色列表（15种颜色）
    customColorList = ['darkorange', "gold", "lawngreen", "lightseagreen", 'purple', 
                    'orange', 'cyan', 'magenta', 'lime', 'pink', 
                    'teal', 'lavender', 'brown', 'beige', 'maroon']

    # 插入黑色作为第一个颜色，对应值0
    colors = ['black'] + customColorList

    # 创建自定义颜色映射
    custom_cmap = ListedColormap(colors)
    
    # 可视化2：聚类热力图
    line , row = 2 , 5

    # fig, axes = plt.subplots(line * 2, row, figsize=fig_size)
    global_min = 0
    # for i , cluster in enumerate(clusters):
    #     h = w = int(math.sqrt(cluster.shape[0]))
    #     cluster_map = cluster.reshape(h, w).astype(float)
    #     axes[int(i / row)][int(i % row)].imshow(cluster_map, cmap=custom_cmap, vmin=global_min , vmax=global_min+15)  # 使用离散颜色映射
    #     axes[int(i / row)][int(i % row)].set_title(f"OHead_{i}")
    #     axes[int(i / row)][int(i % row)].axis('off')

    

    # for i , head in enumerate(pruned_heads):
    #     cluster = translate_head_label_new(head)
    #     cluster_map = cluster.reshape(h, w).astype(float)
    #     assert cluster.min() > 0
    #     print(f"cluster_{i}.min = {cluster.min()} , cluster_{i}.max = {cluster.max()} , cluster_{i}.mask_len = {len(head.get_masks())}")
    #     index = i + len(clusters)
    #     im = axes[int(index / row)][int(index % row)].imshow(cluster_map, cmap=custom_cmap , vmin=global_min , vmax=global_min+15)  # 使用离散颜色映射
    #     # plt.colorbar(im , ticks=np.arange(16), label='Cluster ID')
    #     axes[int(index / row)][int(index % row)].set_title(f"Head_{i}")
    #     axes[int(index / row)][int(index % row)].axis('off')

    fig, axes = plt.subplots(line, row, figsize=fig_size)
    unique_masks = process_masks(pruned_heads)
    assert len(unique_masks) == 10
    for i ,mask in enumerate(unique_masks):
        cluster_map = mask.reshape(h, w).astype(float)
        # assert cluster.min() > 0
        # print(f"cluster_{i}.min = {cluster.min()} , cluster_{i}.max = {cluster.max()} , cluster_{i}.mask_len = {len(head.get_masks())}")
        index = i
        im = axes[int(index / row)][int(index % row)].imshow(cluster_map, cmap=custom_cmap , vmin=global_min , vmax=global_min+15)  # 使用离散颜色映射
        # plt.colorbar(im , ticks=np.arange(16), label='Cluster ID')
        axes[int(index / row)][int(index % row)].set_title(f"mask_{i+1}")
        axes[int(index / row)][int(index % row)].axis('off')
    
    plt.tight_layout()
    file_path = os.path.join(save_dir, f"_decompose_{t}.png")
    fig.savefig(file_path)
    return

def intra_head_ortho(z_ica, masks):
    # ortho features within a head
    ortho_blocks = []
    for mask in masks:
        # 提取区域特征 [n_points,5]
        region_feat = z_ica[mask]
        
        # Gram-Schmidt正交化
        basis = []
        for vec in region_feat.T:
            v_ortho = vec - sum( np.dot(vec, b)*b for b in basis )
            if np.linalg.norm(v_ortho) > 1e-6:
                basis.append(v_ortho / np.linalg.norm(v_ortho))
        
        # 重组为原空间形状
        ortho_block = np.zeros_like(z_ica)
        ortho_block[mask] = np.column_stack(basis)
        ortho_blocks.append(ortho_block)
    
    return np.concatenate(ortho_blocks, axis=1)  # [4096,15]

def merge_overlap_heads(head_data_list  , indep_threshold = 0.9 , iou_threshold = 0.5):
    # all process of merge heads
    n = len(head_data_list)
    redundant_pairs = []
    for i in range(n):
        for j in range(i+1 , n):
            redundant_pair = detect_redundant_masks(head_data_list[i] , head_data_list[j] , indep_threshold , iou_threshold)
            redundant_pairs.extend(redundant_pair)

    pruned_heads = prune_redundant_masks(head_data_list , redundant_pairs)    
    return pruned_heads

def cluster_merge_heads(masked_q ,num_components_from_high_var_q , num_decompose_components,att_matrix_selected, path,t , indep_threshold = 0.9 , iou_threshold = 0.5 , visualise = True):
    head_data_list ,cluster_label_list = cluster_high_var_q(masked_q , num_components_from_high_var_q , num_decompose_components , att_matrix_selected)
    pruned_heads = merge_overlap_heads(head_data_list)
        # ortho_features = intra_head_ortho(S, area_masks)
    if visualise:
        visualise_cluster_masks(cluster_label_list , pruned_heads , path , t)
    return pruned_heads

def gen_corred_comps(mask_id , head_list):
    comp_dict = {}
    for head in head_list:
        comp_index = head.find_independent_comp_by_mask_id(mask_id)
        if comp_index != -1:
            print(f"head_{head.head_id}found:comp!!")
            comp_dict.update({f"head_{head.get_head_id}":comp_index})
            head.reconstruct_data(head.get_ica_comp().T)
    return comp_dict

def save_masks(path , heads):
    os.makedirs(path, exist_ok=True)
    for i , head in enumerate(heads):
        file_path = os.path.join(path , f"mask_head_{i}.pt")
        cluster = translate_head_label_new(head)
        torch.save(cluster , file_path)
    print(f"done save all masks to {path}")

def gen_mask_flags(config , config_path):
    OmegaConf.update(config , "compute_mask" , True)
    OmegaConf.update(config , "masks" , [1])
    OmegaConf.update(config , "prompt" , "")

    with open(config_path , "w") as f:
        OmegaConf.save(config , f)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_dir",
        type=str,
        nargs="?",
        default="/data/zsc/feature_decompose_diffusion/latent-diffusion/feature_dir/sdv1.4/samples/00470000",
        help="path to feature maps , where settles the feature config"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="models/ldm/sdv1.4/config.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/sdv1.4/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--mask_time",
        type=int,
        default=1,
        help="time to save mask",
    )

    opt = parser.parse_args()
    model_config = OmegaConf.load(f"{opt.model_config}")
    model, _ = load_model(model_config, opt.ckpt, True, True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    num_components_from_high_var_q = 10 # decompose 5 main components from high variance q
    num_decompose_components = 5 # decompose 3 main components from each components of high variance q

    feature_types = [
                    # "in_layers_features",
                    # "out_layers_features",
                    "self_attn_q",
                    # "self_attn_k",
                    # "self_attn_v",
                ]
    setup_config = OmegaConf.load("./configs/setup.yaml")


    for root , dirs , files in os.walk(opt.feature_dir):
        for dir in sorted(dirs):
            mask_config = os.path.join(root , dir , "mask_config.yaml")
            if os.path.exists(mask_config):

                exp_path_root = os.path.join(root , dir)#setup_config.config.exp_path_root#featire_dir
                exp_config = OmegaConf.load(mask_config)
                transform_experiments = exp_config.config.experiments_transform
                fit_experiments = exp_config.config.experiments_fit
                exp_info_dir = exp_config.config.experiments_info_dir
                ddim_config = OmegaConf.load(os.path.join(exp_info_dir[0], "sampling_config.yaml"))
                ddim_steps = ddim_config.custom_steps
                if 'compute_mask' in exp_config and exp_config.compute_mask:
                    continue

                
                sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0, verbose=False)
                time_range = np.flip(sampler.ddim_timesteps)
                print(f"put ddim_step = {ddim_steps} and get sampler.ddim_step = {sampler.ddim_timesteps}")
                total_steps = sampler.ddim_timesteps.shape[0]
                iterator = tqdm(time_range, desc="visualizing features", total=total_steps)

                # print(f"visualizing features PCA experiments: block - {exp_config.config.block}; transform experiments - {exp_config.config.experiments_transform}; fit experiments - {exp_config.config.experiments_fit}")

                transform_feature_maps_paths = []
                for experiment in transform_experiments:
                    transform_feature_maps_paths.append(os.path.join(experiment, "feature_maps"))

                fit_feature_maps_paths = []
                for experiment in fit_experiments:
                    fit_feature_maps_paths.append(os.path.join(experiment, "feature_maps"))

                
                # feature_pca_paths = {}

                # pca_folder_path = os.path.join(exp_path_root, "PCA_features_vis", exp_config.config.experiment_name)
                # os.makedirs(pca_folder_path, exist_ok=True)

                # for feature_type in feature_types:
                #     feature_pca_path = os.path.join(exp_path_root, f"{exp_config.config.block}_{feature_type}")
                #     feature_pca_paths[feature_type] = feature_pca_path
                #     os.makedirs(feature_pca_path, exist_ok=True)

                for t in iterator:
                    if t != opt.mask_time:
                        continue
                    for feature_type in feature_types:
                        mask_path = os.path.join(exp_path_root , f"mask_{opt.mask_time}")
                        fit_features , fit_origin = load_experiments_features(fit_feature_maps_paths, exp_config.config.block, feature_type, t)  # N X C
                        att_matrix = load_experiments_att_matrix(fit_feature_maps_paths, exp_config.config.block, feature_type, t)
                        att_matrix_tensor = torch.cat(att_matrix , dim = 0)
                        att_matrix_selected = att_matrix_tensor[0]
                        all_head = cluster_merge_heads(torch.cat(fit_origin , dim = 0) , num_components_from_high_var_q , num_decompose_components , att_matrix_selected , exp_path_root , t)
                        save_masks(mask_path , all_head)
                        gen_mask_flags(exp_config , mask_config)

            else:
                print(f"no mask_config.yaml in {dir} , skip this dir")
                continue

if __name__ == "__main__":
    main()
