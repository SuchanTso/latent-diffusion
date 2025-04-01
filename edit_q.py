import argparse, os
# from tqdm import trange
import torch
from einops import rearrange
from omegaconf import OmegaConf
import json
# from feature_extraction import load_model
import numpy as np
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib.pyplot as plt
from ddim_process import *


from decompose_util import detect_redundant_masks , prune_redundant_masks , translate_head_label_new , cluster_high_var_q , edit_head_data


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

    return pruned_heads # return heads for contineous process

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="configs/ddim_edit/edit.yaml",
        help="path to the ddim edit config file"
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
    return parser

def gen_init_img_latent(model , image_path , device):
    init_image = load_init_image(image_path , device)
    init_latent = load_init_latent(model , init_image)
    return init_image , init_latent


def gen_editted_dir(sampler , model , z_T , editted_data , t ,block_num ,save_path,  custom_steps = 50):
    noise_list = []
    def fetch_specified_step_callback(noise):
        # print(f"fetch_specified_step_callback , noise.shape = {noise.shape}")
        noise_list.append(noise)
    ori_no_valid_sampling = ddim_process_sampling(model , sampler , z_T ,
                                                  custom_steps ,
                                                  gen_prompt(model , ""), batch_size=1 , eta=0.0 , verbose=False,
                                                  noise_extract_step=t , noise_extract_callback=fetch_specified_step_callback)
    # assert len(editted_data) == 3
    # edit_data = list(editted_data.values())[0]
    injected_features = editted_data#[editted_data] * custom_steps
    # injected_features_tensor = torch.cat(injected_features , dim=0).to(model.device())
    editted_no_valid_sampling = ddim_process_sampling(model , sampler , z_T ,
                                                  custom_steps ,
                                                  gen_prompt(model , ""), batch_size=1 , eta=0.0 , verbose=False,
                                                  injected_features=injected_features,
                                                  noise_extract_step=t , noise_extract_callback=fetch_specified_step_callback)
    assert len(noise_list) == 2 , f"len(noise_list) = {len(noise_list)}"
    editted_dir = noise_list[1] - noise_list[0]
    print(f"diff.range = [{editted_dir.min()} , {editted_dir.max()}]")
    img_path = os.path.join(save_path, f"injected_.png")
    save_samples(img_path , editted_no_valid_sampling)
    return editted_dir


def main():
    parser = arg_parse()
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
    
    image , x_encode = gen_init_img_latent(model , exp_config.config.img_path , device)

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
    head_data_in_time = None
    for t in iterator:
        if t != exp_config.config.edit_time_step:
            #TODO: save these head data for next step @Suchan
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
            feature_heads = cluster_merge_heads(torch.cat(fit_origin , dim = 0) , num_components_from_high_var_q , num_decompose_components , att_matrix_selected , feature_pca_paths[feature_type] , t , visualise = False)
            if t == exp_config.config.edit_time_step:
                head_data_in_time = feature_heads
                # cluster_masks[head_key] = masks
            # merge_feat = auto_ortho_merge(componets_dic, mask_dic)
            # print(f"merge_feat[0].shape = {merge_feat[0].shape}")
    start_code_path = os.path.join(exp_config.config.save_dir , "start_code.pt") if exp_config.config.use_start_code else ""
    z_T , sampler_ = ddim_inversion(model , x_encode , None , device , start_code_path=start_code_path)
    if exp_config.config.use_start_code:
        torch.save(z_T , start_code_path)

    unet_model = model.model.diffusion_model

    editt_data_ = {}
    editt_data_list = []
    def specified_step_callback(i):
        # if i >= exp_config.config.edit_time_step:
        #     print(f"specified_step_callback_step = {i}")
        #     editt_data_list.append(None)
        #     return
        block_str = exp_config.config.block
        block_type , block_ss , block_num = block_str.split("_")
        # block_num = 4
        attn_block_nums = [4,5,6,7,8,9,10,11]
        res_block_nums =[4]
        if block_type == "output":
            to_extract_blcok = unet_model.output_blocks
        else:
            to_extract_blcok = unet_model.input_blocks
        for block_num in attn_block_nums:
            
            attn_q_data = extract_specific_block_features(to_extract_blcok , block_num , feature_type="self_attn_q")
            attn_k_data = extract_specific_block_features(to_extract_blcok , block_num , feature_type="self_attn_k")
            q_feature_key = f'output_block_{block_num}_self_attn_q'
            k_feature_key = f'output_block_{block_num}_self_attn_k'

            # print(f"res_data.shape = {res_data.shape} , attn_q_data.shape = {attn_q_data.shape} , attn_k_data.shape = {attn_k_data.shape}")
            assert attn_q_data is not None , f"block_num:{block_num}not find self_attn_q"
            mask_id = 3
            editt_data_.update({q_feature_key:attn_q_data})
            editt_data_.update({k_feature_key:attn_k_data})
            # if block_num == 9 and i == exp_config.config.edit_time_step:
            #     editted_data = edit_head_data(head_data_in_time , attn_q_data , mask_id)
            #     editt_data_.update({q_feature_key:editted_data})


        for block_num in res_block_nums:
            out_layers_feature_key = f'output_block_{block_num}_out_layers'
            res_data = extract_specific_block_features(to_extract_blcok , block_num , feature_type="ResBlock")
            editt_data_.update({out_layers_feature_key:res_data})#editted_data, data for test
        editt_data_list.append(editt_data_)
        # diff = editted_data - attn_q_data.cpu()
        # print(f"got into specified_step_callback , i = {i} , diff.range = [{diff.min()} , {diff.max()}]")
        
        
    def extract_specific_block_features(blocks ,block_num , feature_type="self_attn_q"):
        block_idx = 0
        # print(f"blocks.len = {len(blocks)}")
        for i , block in enumerate(blocks):
            # print(f"block_idx = {block_idx} , block_num = {block_num} ")
            if block_idx == int(block_num):
                if feature_type == "self_attn_q":
                    # assert feature_type == "self_attn_q" and "SpatialTransformer" in str(type(block[1]))
                    assert block[1].transformer_blocks[0].attn1.q is not None
                    return block[1].transformer_blocks[0].attn1.q
                elif feature_type == "ResBlock":
                # assert feature_type == "self_attn_q" and "SpatialTransformer" in str(type(block[1]))
                    # assert block[1].transformer_blocks[0].attn1.q is not None
                    return block[0].out_layers_features
                elif feature_type == "self_attn_k":
                    return block[1].transformer_blocks[0].attn1.k
            else:
                block_idx += 1
            
        assert False , f"iterator done , didn't get {block_num} while len(blocks) = {len(blocks) } and block_idx = {block_idx}"

    no_valid_sample = ddim_process_sampling(model , sampler_ , z_T , ddim_steps , gen_prompt(model , exp_config.config.prompt), batch_size=1 , eta=0.0 , verbose=False,callback_ddim_timesteps=None , callback=specified_step_callback)
    print(f"after call back here")
    assert len(editt_data_) > 0 , "editt_data_ is None"
    _ , _ , block_num = exp_config.config.block.split("_")
    editted_dir = gen_editted_dir(sampler_ , model , z_T , editt_data_list  , exp_config.config.edit_time_step , block_num , save_path=exp_config.config.save_dir ,custom_steps=ddim_steps)
    editted_img = ddim_process_sampling(model , sampler_ , z_T ,
                                        ddim_steps ,
                                        gen_prompt(model , ""), 
                                        batch_size=1 , eta=0.0 , verbose=False,
                                        edit_scale=1.,
                                        edit_dir=editted_dir)
    os.makedirs(exp_config.config.save_dir ,exist_ok=True)
    save_path = exp_config.config.save_dir
    save_samples(save_path , editted_img)
    save_samples(os.path.join(exp_config.config.save_dir , "prompt.png") , no_valid_sample)
    
    # save_samples(os.path.join(exp_config.config.save_dir , "noise_vec.png") , editted_dir)





if __name__ == "__main__":
    main()
