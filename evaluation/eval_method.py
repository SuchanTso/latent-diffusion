import lpips
import torch
import numpy as np
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial

# def lpips_eval(input_img_path , output_img_path , loss_fn):
#     """
#     evaluate lpips score
#     """
#     if os.path.exists(input_img_path) and os.path.exists(output_img_path):
#         img0 = lpips.im2tensor(lpips.load_image(input_img_path))
#         img1 = lpips.im2tensor(lpips.load_image(output_img_path))
#         current_lpips_distance = loss_fn.forward(img0 , img1)
#         return current_lpips_distance
#     else:
#         print(f"input_img_path = {input_img_path} or output_img_path = {output_img_path} not exist")
#         return 0.0
    
# def calculate_clip_score(image:np.array, prompts:str , clip_score_fn):
#     # import pdb;pdb.set_trace()
#     # images_int = (np.asarray(images[0]) * 255).astype("uint8")
#     image_int = ((np.asarray(image) * 255).astype("uint8"))[None, ...]
#     clip_score = clip_score_fn(torch.from_numpy(image_int).permute(0, 3, 1, 2), prompts).detach()
#     return round(float(clip_score), 4)

# def calculate_clip_score_path(image_path :str, prompts:str , clip_score_fn):
#     # import pdb;pdb.set_trace()
#     # images_int = (np.asarray(images[0]) * 255).astype("uint8")
#     print(f"image_path = {image_path}")
#     image = Image.open(image_path).convert("RGB")
#     return calculate_clip_score(image , prompts , clip_score_fn)

class EvalController:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pass
    def load_lpips(self):
        self.loss_fn = lpips.LPIPS(net='alex' , version='0.1')
        self.loss_fn.to(self.device)

    def load_clip_model(self , path):
        self.clip_model = partial(clip_score, model_name_or_path=path)

    
    def calculate_clip_score(self , image_path , prompts):
        assert self.clip_model is not None , "please load clip model first"
        image = Image.open(image_path).convert("RGB")
        image_int = ((np.asarray(image) * 255).astype("uint8"))[None, ...]
        clip_score = self.clip_model(torch.from_numpy(image_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)
    
    def calculate_lpips_score(self , input_img_path , output_img_path):
        img0 = lpips.im2tensor(lpips.load_image(input_img_path)).to(self.device)
        img1 = lpips.im2tensor(lpips.load_image(output_img_path)).to(self.device)
        current_lpips_distance = self.loss_fn.forward(img0 , img1)
        return current_lpips_distance


in_path = '/data/zsc/feature_decompose_diffusion/latent-diffusion/edit_lab/000000000024/in/ddim.png'
out_path = '/data/zsc/feature_decompose_diffusion/latent-diffusion/edit_lab/000000000024/out/sample_0.png'