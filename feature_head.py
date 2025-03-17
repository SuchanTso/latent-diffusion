import argparse, os
from tqdm import trange
import torch
from einops import rearrange
from pnp_utils import visualize_and_save_features_pca , visualize_and_save_features_dbscan , visualize_and_save_features_kmeans , decompose_q_high_var_ica , decompose_spectral
from omegaconf import OmegaConf
import json
from feature_extraction import load_model
import numpy as np
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt


from decompose_util import  encode_head_data , detect_redundant_masks , prune_redundant_masks , translate_head_label_new , cluster_high_var_q


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
    line , row = 1 , 8

    global_min = 0
    fig, axes = plt.subplots(line * 2, row, figsize=fig_size)
    for i , cluster in enumerate(clusters):
        cluster_map = cluster.reshape(h, w).astype(float)
        axes[int(i / row)][int(i % row)].imshow(cluster_map, cmap=custom_cmap, vmin=global_min , vmax=global_min+15)  # 使用离散颜色映射
        axes[int(i / row)][int(i % row)].set_title(f"OCluster_{i}")
        axes[int(i / row)][int(i % row)].axis('off')

    for i , head in enumerate(pruned_heads):
        cluster = translate_head_label_new(head)
        cluster_map = cluster.reshape(h, w).astype(float)
        assert cluster.min() > 0
        print(f"cluster_{i}.min = {cluster.min()} , cluster_{i}.max = {cluster.max()} , cluster_{i}.mask_len = {len(head.get_masks())}")
        index = i + len(clusters)
        im = axes[int(index / row)][int(index % row)].imshow(cluster_map, cmap=custom_cmap , vmin=global_min , vmax=global_min+15)  # 使用离散颜色映射
        # plt.colorbar(im , ticks=np.arange(16), label='Cluster ID')
        axes[int(index / row)][int(index % row)].set_title(f"Cluster_{i}")
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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="configs/feature_vis/feature-pca-vis.yaml",
        help="path to the feature PCA visualization config file"
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

    opt = parser.parse_args()
    setup_config = OmegaConf.load("./configs/setup.yaml")
    exp_path_root = setup_config.config.exp_path_root
    exp_config = OmegaConf.load(f"{opt.config}")
    transform_experiments = exp_config.config.experiments_transform
    fit_experiments = exp_config.config.experiments_fit
    exp_info_dir = exp_config.config.experiments_info_dir

    # with open(os.path.join(exp_path_root, transform_experiments[0], "args.json"), "r") as f:
    #     args = json.load(f)
    #     ddim_steps = args["save_feature_timesteps"][-1]
    # load ddim config ,aka time step
    ddim_config = OmegaConf.load(os.path.join(exp_info_dir[0], "sampling_config.yaml"))
    ddim_steps = ddim_config.custom_steps

    model_config = OmegaConf.load(f"{opt.model_config}")
    model, _ = load_model(model_config, opt.ckpt, True, True)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    print(f"put ddim_step = {ddim_steps} and get sampler.ddim_step = {sampler.ddim_timesteps}")
    total_steps = sampler.ddim_timesteps.shape[0]
    iterator = tqdm(time_range, desc="visualizing features", total=total_steps)

    print(f"visualizing features PCA experiments: block - {exp_config.config.block}; transform experiments - {exp_config.config.experiments_transform}; fit experiments - {exp_config.config.experiments_fit}")

    transform_feature_maps_paths = []
    for experiment in transform_experiments:
        transform_feature_maps_paths.append(os.path.join(experiment, "feature_maps"))

    fit_feature_maps_paths = []
    for experiment in fit_experiments:
        fit_feature_maps_paths.append(os.path.join(experiment, "feature_maps"))

    feature_types = [
        # "in_layers_features",
        # "out_layers_features",
        "self_attn_q",
        # "self_attn_k",
        # "self_attn_v",
    ]
    feature_pca_paths = {}

    pca_folder_path = os.path.join(exp_path_root, "PCA_features_vis", exp_config.config.experiment_name)
    os.makedirs(pca_folder_path, exist_ok=True)

    for feature_type in feature_types:
        feature_pca_path = os.path.join(pca_folder_path, f"{exp_config.config.block}_{feature_type}")
        feature_pca_paths[feature_type] = feature_pca_path
        os.makedirs(feature_pca_path, exist_ok=True)

    num_components_from_high_var_q = 5 # decompose 5 main components from high variance q
    num_decompose_components = 5 # decompose 3 main components from each components of high variance q
    for t in iterator:
        if t >= 20:
            continue
        for feature_type in feature_types:
            fit_features , fit_origin = load_experiments_features(fit_feature_maps_paths, exp_config.config.block, feature_type, t)  # N X C
            # haed_features = load_experiments_heads(fit_feature_maps_paths, exp_config.config.block, feature_type, t)
            masked_q , mask = spatial_variance_filter(torch.cat(fit_origin , dim = 0))
            att_matrix = load_experiments_att_matrix(fit_feature_maps_paths, exp_config.config.block, feature_type, t)
            att_matrix_tensor = torch.cat(att_matrix , dim = 0)
            # att_matrix_selected = att_matrix_tensor[0][mask]
            att_matrix_selected = att_matrix_tensor[0]
            print(f"att_matrix_selected.shape = {att_matrix_selected.shape}")
            # print(f"fit_features_origin.shape = {torch.cat(fit_origin , dim = 0).shape}")
            # print(f"masked_q.shape = {masked_q.shape}")
            cluster_merge_heads(torch.cat(fit_origin , dim = 0) , num_components_from_high_var_q , num_decompose_components , att_matrix_selected , feature_pca_paths[feature_type] , t)
                # cluster_masks[head_key] = masks
            # merge_feat = auto_ortho_merge(componets_dic, mask_dic)
            # print(f"merge_feat[0].shape = {merge_feat[0].shape}")


if __name__ == "__main__":
    main()
