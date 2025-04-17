import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from edit_q import load_unique_mask , resize_mask
from omegaconf import OmegaConf


def overlay_mask_on_image(img, mask, output_path, mask_color=(0.8, 0.3, 0.2), alpha=0.9):
    """
    将 mask 叠加到原图上并保存图像。
    
    参数：
    - image_path: str, 原图文件路径。
    - mask: 2D numpy array, 提取的 mask（与原图尺寸相同，值为 0 或 1）。
    - output_path: str, 保存结果图像的路径。
    - mask_color: tuple, mask 的颜色 (R, G, B)，范围 [0, 1]。
    - alpha: float, mask 的透明度，范围 [0, 1]。
    """
    # 加载原图
    image = img / 255.0  # 转换为 [0, 1] 范围的浮点数
    
    # 确保 mask 是 2D 数组
    if mask.ndim != 2:
        raise ValueError(f"mask 必须是二维数组 ,shape = {mask.shape}")
    
    # 创建一个与原图大小相同的彩色 mask
    color_mask = np.zeros_like(image)
    for i in range(3):  # 遍历 RGB 通道
        color_mask[:, :, i] = mask * mask_color[i]
    
    # 叠加 mask 到原图
    overlayed_image = image * (1 - mask[:, :, np.newaxis]) + color_mask * alpha + image * (1 - alpha)
    return overlayed_image
    # # 显示并保存结果
    # plt.figure(figsize=(10, 10))
    # plt.imshow(overlayed_image)
    # plt.axis('off')  # 关闭坐标轴
    # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # 保存图像
    # plt.close()

def count_num_bit(number):
    cnt = 0
    while number != 0:
        cnt+=1
        number //= 10
    return cnt

def load_all_masks(mask_path , shape):
    mask_list = []
    for i in range(8):
        mask = load_unique_mask(mask_dir=mask_path , block_type='output_block_9',mask_id=[i])
        if mask is not None:
            mask = resize_mask(mask , shape)
            mask_list.append(mask)
    return mask_list

def vis_merge_list(img_list , out_path , prompt , mask_list):
    line = 3
    row = 4
    fig_size = (32 , 32)
    fig, axes = plt.subplots(line, row, figsize=fig_size)
    for i ,img in enumerate(img_list):
        index = i
        if i >= len(img_list) - 2:
            idx = 1 - (len(img_list) - i - 1)
            title = prompt if idx == 0 else f"{mask_list}"
            im = axes[int(2)][idx].imshow(img)  # 使用离散颜色映射
            # plt.colorbar(im , ticks=np.arange(16), label='Cluster ID')
            axes[int(2)][idx].set_title(title , fontsize = 32)
            axes[int(2)][idx].axis('off')
        else:
            im = axes[int(index / row)][int(index % row)].imshow(img)  # 使用离散颜色映射
            # plt.colorbar(im , ticks=np.arange(16), label='Cluster ID')
            axes[int(index / row)][int(index % row)].set_title(f"mask_{i+1}", fontsize = 32)
            axes[int(index / row)][int(index % row)].axis('off')
    
    plt.tight_layout()
    fig.savefig(out_path)

def main():
    img_list = [118,176,501,685,932,936,1109,1032,1435,327,433,630,633]
    full_bit = 12
    img_dir = '/data/zsc/feature_decompose_diffusion/latent-diffusion/edit_lab'
    mask_dir = '/data/zsc/feature_decompose_diffusion/latent-diffusion/feature_dir/sdv1.4/samples/00470000'
    # 原图路径
    for img_idx in img_list:
        zero_num = full_bit - count_num_bit(img_idx)
        file_name = '0'*zero_num +str(img_idx)
        image_path = os.path.join(img_dir , file_name ,'in','ddim.png' )
        image = Image.open(image_path)
        image_array = np.array(image)

        print("图像形状:", image_array.shape)
        print(image_path)
        mask_path = os.path.join(mask_dir , file_name)
        config_path = os.path.join(mask_path , 'mask_config.yaml')
        config = OmegaConf.load(config_path)
        prompt_str = config.prompt
        mask_list = config.masks

        # 假设有一个 mask，形状与原图一致
        
        masks = load_all_masks(mask_path , shape = (image_array.shape[0] , image_array.shape[1]))

        
        # 输出路径
        output_path = os.path.join('/data/zsc/feature_decompose_diffusion/latent-diffusion/vis' , f"output_overlayed_image_{img_idx}.png")
        
        # 调用函数
        merge_list = []
        for mask in masks:
            merged_img = overlay_mask_on_image(image_array, mask[0][0].numpy(), output_path, mask_color=(0.8,0.3,0.2), alpha=0.5)
            merge_list.append(merged_img)

        prompt_img_path = image_path = os.path.join(img_dir , file_name ,'out','prompt.png' )    
        edit_img_path = image_path = os.path.join(img_dir , file_name ,'out','sample_0.png' ) 
        prompt = np.array(Image.open(prompt_img_path)) / 255.0
        edit = np.array(Image.open(edit_img_path)) / 255.0
        merge_list.append(prompt)
        merge_list.append(edit)  
        vis_merge_list(merge_list , output_path , prompt_str , mask_list)

# 示例用法
if __name__ == "__main__":
    main()
    