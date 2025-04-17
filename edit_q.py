import argparse, os
# from tqdm import trange
import torch
from omegaconf import OmegaConf
# from feature_extraction import load_model
import numpy as np

from ldm.models.diffusion.ddim import DDIMSampler
from ddim_process import *
import math
import torch.nn.functional as F
from torchvision.transforms import functional

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

def resize_masks(masks , resolution):
    """
    resize mask to any resolution
    mask: shape of (len , ) . assure that sqrt(len) is a precise number , mask sure that mask is binary mask
    resolution: (w , h)
    """
    mask_resize_tensor = []
    print(f"masks.shape = {masks.size()[0]}")
    for mask in masks:
        mask_resized = resize_mask(mask , resolution)
    mask_resize_tensor.append(mask_resized)
    return torch.cat(mask_resize_tensor , dim = 0)

def resize_mask(mask , resolution):
    assert mask.min() == 0 and mask.max() <= 1 , f"got non-binary mask , range from [{mask.min()} , {mask.max()}]"
    h = w = int(math.sqrt(mask.size()[0]))
    assert h * w == mask.size()[0] , f"mask is shape of {mask.shape} that is illeagal"
    width , height = resolution
    mask_reshape = mask.reshape(h , w).unsqueeze(0).unsqueeze(0)
    # print(f"mask_reshape = {mask_reshape.shape}")
    mask_resized = F.interpolate(
    mask_reshape, 
    size=(int(height) , int(width)), 
    mode='bicubic'  # 对于二值化 mask 推荐使用 nearest
    )
    return mask_resized

def load_mask(mask_dir  , mask_id_list = [1] , mask_time = 1):
    mask_path = os.path.join(mask_dir , f"mask_{mask_time}")
    print(f"loading mask_{mask_id_list} in {mask_path}")
    masks_norm_list = []
    for _, dirs, files in os.walk(mask_path):
        for i , file in enumerate(files):
            file_path = os.path.join(mask_path , f"mask_head_{i}.pt")
            masks = torch.tensor(torch.load(file_path))
            masks_norm = torch.zeros_like(masks)
            for mask_id in mask_id_list:
                masks_norm |= (masks == mask_id)
            masks_norm_list.append(masks_norm.float())
        break
    print(f"done loading {len(masks_norm_list)}")
    return torch.stack(masks_norm_list , dim=0)

def load_unique_mask(mask_dir , block_type , mask_id = [1] , mask_time = 1):
    masks = load_mask(mask_dir , mask_id , mask_time)
    for mask in masks:
        if mask.max() > 0:
            return mask
    return None
    # assert False , f"mask_id = {mask_id} not found in {mask_dir} , block_type = {block_type}, mask_time = {mask_time}"


def get_x_T_i(model , time_range , res_list , edit_time):
    for i , time in enumerate(time_range):
        if time == edit_time:
            assert i + 1 < len(res_list) , f"i + 1 = {i + 1} is out of range of res_list"
            x_T_i = model.decode_first_stage(res_list[i + 1])
            return x_T_i
    assert False , f"edit_time = {edit_time} is not in time_range = {time_range}"

def main():
    parser = arg_parse()
    opt = parser.parse_args()
    exp_config = OmegaConf.load(f"{opt.config}")
    mask_dir = exp_config.config.mask_dir_root
    feature_dir = exp_config.config.feature_root
    model_config = OmegaConf.load(f"{opt.model_config}")
    model, _ = load_model(model_config, opt.ckpt, True, True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    for root , dirs , files in os.walk(mask_dir):
        # print(f"root = {root} , dirs = {dirs} , files = {files}")
        for dir in dirs:
            feature_fold = os.path.join(feature_dir , dir)
            mask_config_file = os.path.join(feature_fold , "mask_config.yaml")
            if os.path.exists(mask_config_file):
                mask_config = OmegaConf.load(mask_config_file)
                if 'prompt'not in mask_config or len(mask_config.prompt) == 0:
                    print(f"{feature_fold} is not edittable")
                    continue
                exp_info_dir = mask_config.config.experiments_info_dir
                ddim_config = OmegaConf.load(os.path.join(exp_info_dir[0], "sampling_config.yaml"))#no
                ddim_steps = ddim_config.custom_steps

                sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0, verbose=False)
                time_range = np.flip(sampler.ddim_timesteps)
                print(f"put ddim_step = {ddim_steps} and get sampler.ddim_step = {sampler.ddim_timesteps}")
                total_steps = sampler.ddim_timesteps.shape[0]

                # image , x_encode = gen_init_img_latent(model , exp_config.config.img_path , device)
                start_code_path = os.path.join(feature_fold , "img" , "start_pt.pt")
                z_T , sampler_ = ddim_inversion(model , None , None , device , start_code_path=start_code_path)

                no_valid_sample , origin_intermediate = ddim_process_sampling(model , sampler_ , z_T , ddim_steps , gen_prompt(model , ""), batch_size=1 , eta=0.0 , verbose=False,callback_ddim_timesteps=None)
                print(f"origin_intermediate.shape ={len(origin_intermediate['x_inter'])} , time = {origin_intermediate['time_range']}")
                block_str = mask_config.config.block
                mask_fold = os.path.join(mask_dir , dir)
                mask_q = load_unique_mask(mask_fold , block_str , mask_id = mask_config.masks)
                def blend_mask_noise(z_T_i , i):
                    cur_time = origin_intermediate['time_range'][i]
                    if cur_time != exp_config.config.edit_time_step:
                        # print(f"cur_time = {cur_time} , not equal to edit_time = {exp_config.config.edit_time_step}")
                        return z_T_i
                    y_T_i = model.decode_first_stage(z_T_i)
                    x_T_i = get_x_T_i(model , origin_intermediate['time_range'] , origin_intermediate['x_inter'] , exp_config.config.edit_time_step).to(device)
                    mask = resize_mask(mask_q , (y_T_i.shape[2] , y_T_i.shape[3])).to(device)
                    x_T_i_new = y_T_i * mask + x_T_i * (1 - mask)
                    z_T_i_new = model.get_first_stage_encoding(model.encode_first_stage(x_T_i_new))
                    # print(f"blend_mask")
                    return z_T_i_new

                editted_img , _= ddim_process_sampling(model , sampler_ , z_T ,
                                                    ddim_steps ,
                                                    gen_prompt(model , mask_config.prompt), 
                                                    batch_size=1 , eta=0.0 , verbose=False,
                                                    blend_callback = blend_mask_noise
                                                    )
                prompted_img , _= ddim_process_sampling(model , sampler_ , z_T ,
                                                    ddim_steps ,
                                                    gen_prompt(model , mask_config.prompt), 
                                                    batch_size=1 , eta=0.0 , verbose=False
                                                    )
                print(f"mask_q.shape = {mask_q.shape} , sample.shape = {editted_img.shape} , z_T.shape = {z_T.shape}")
                os.makedirs(exp_config.config.save_dir ,exist_ok=True)
                save_path = os.path.join(exp_config.config.save_dir , dir)
                save_path_in = os.path.join(save_path , "in")
                save_path_out = os.path.join(save_path , "out")

                os.makedirs(save_path , exist_ok=True)
                os.makedirs(save_path_out , exist_ok=True)
                os.makedirs(save_path_in, exist_ok=True)
                
                save_samples(save_path_out, editted_img)
                save_samples(os.path.join(save_path_out, "prompt.png") , prompted_img)
                save_samples(os.path.join(save_path_in, "ddim.png") , no_valid_sample)
            else:
                print(f"mask_config.yaml not found in {feature_fold} , skip this fold")
                continue
        break


if __name__ == "__main__":
    main()
