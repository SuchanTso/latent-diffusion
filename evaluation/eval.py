from eval_method import EvalController
import argparse
import os
import time
from omegaconf import OmegaConf

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log_dir",
        type=str,
        nargs="?",
        default="/data/zsc/feature_decompose_diffusion/latent-diffusion/feature_dir/eval",
        help="path to record score"
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default="/data/zsc/feature_decompose_diffusion/latent-diffusion/edit_lab",
        help="path to the input image"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="/data/zsc/feature_decompose_diffusion/latent-diffusion/feature_dir/sdv1.4/samples/00470000",
        help="path to the prompt dir"
    )
    return parser

def gen_default_record(log_path):
    """
    generate default record
    """
    record = {
        "lpips": [],
        "clip_score": [],
        "imp_cnt":0
    }
    with open(log_path , 'w') as f:
        OmegaConf.save(config=record , f=f)

def get_prompt_config(prompt_path):
    """
    get prompt config
    """
    assert os.path.exists(prompt_path)
    config = OmegaConf.load(prompt_path)
    prompt = config.prompt
    return prompt

def main():
    parser = arg_parse()
    opt = parser.parse_args()
    now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    log_file = os.path.join(opt.log_dir , f"eval_{now}.yaml")
    gen_default_record(log_file)

    eval_controller = EvalController()
    clip_model = '/data/zsc/feature_decompose_diffusion/latent-diffusion/openai/clip-vit-large-patch14'
    eval_controller.load_lpips()
    eval_controller.load_clip_model(clip_model)
    img_cnt = 0
    lpips_score = 0
    clip_score = 0
    for root , dirs , files in os.walk(opt.img_path):
        print(f"dirs = {dirs}")
        for dir in dirs:
            input_img_path = os.path.join(root , dir , "in" , "ddim.png")
            output_img_path = os.path.join(root , dir , "out" , "sample_0.png")
            prompt_file_config = os.path.join(opt.prompt_path , dir , "mask_config.yaml")
            if os.path.exists(input_img_path) and os.path.exists(output_img_path):
                prompt = get_prompt_config(prompt_file_config)
                lpips_score += eval_controller.calculate_lpips_score(input_img_path , output_img_path)
                clip_score += eval_controller.calculate_clip_score(input_img_path , prompt)
                img_cnt += 1
                pass
            else:
                print(f"input_img_path = {input_img_path} or output_img_path = {output_img_path} not exist")
        break
    # print(f"img_cnt = {img_cnt} , lpips_score = {lpips_score} , clip_score = {clip_score}")
    if img_cnt > 0:
        lpips_score /= img_cnt
        clip_score /= img_cnt
    else:
        lpips_score = 0
        clip_score = 0
    print(f"img_cnt = {img_cnt} , lpips_score = {lpips_score} , clip_score = {clip_score}")
    record = {
        "lpips": lpips_score,
        "clip_score": clip_score,
        "img_cnt": img_cnt
    }
    with open(log_file , 'w') as f:
        OmegaConf.save(config=record , f=f)

if __name__ == "__main__":
    main()