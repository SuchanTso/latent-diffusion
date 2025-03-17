import os
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import transforms as T
from math import sqrt
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image
from sklearn.cluster import DBSCAN , KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# load safety model
safety_model_id = "models/sd_safety_checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def visualize_and_save_features_pca(feature_maps_fit_data,feature_maps_transform_data, transform_experiments, t, save_dir):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(len(transform_experiments), -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(transform_experiments):
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        # print(f"pca_img.shape = {pca_img.shape}")
        h = w = int(sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
        file_path = os.path.join(save_dir, f"_time_{t}.png")
        print(file_path)
        pca_img.save(file_path)

def decompose_spectral(combined_data , n_components):
    from sklearn.cluster import SpectralClustering
    spectral_clustering = SpectralClustering(n_clusters=n_components,
                                         affinity='nearest_neighbors',
                                         random_state=0)
    # 使用fit_predict方法对数据进行拟合并预测每个样本所属的簇标签
    cluster_labels = spectral_clustering.fit_predict(combined_data)
    return cluster_labels

def decompose_q_high_var_ica(q , n_components = 3):
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=n_components)
    S_ = ica.fit_transform(q.cpu().numpy())  # ICA components
    A_ = ica.mixing_  # estimated mixing matrix
    return S_ , A_

def visualize_and_save_features_dbscan(feature_maps_fit_data, feature_maps_transform_data, 
                                       transform_experiments, t, save_dir, eps=15.0, min_samples=10):
    """
    对特征图数据进行 DBSCAN 聚类，并将结果保存为图像。
    
    参数：
        - feature_maps_fit_data: 用于 DBSCAN 拟合的特征图数据 (tensor or numpy array)。
        - feature_maps_transform_data: 待转换的特征图数据 (tensor or numpy array)。
        - transform_experiments: 对应的实验数据，控制批次处理 (list)。
        - t: 时间步，用于保存文件名。
        - save_dir: 保存路径。
        - eps: DBSCAN 的邻域半径。
        - min_samples: DBSCAN 的最小样本数。
    
    输出：
        - 将聚类结果可视化为图像并保存。
    """
    # 将张量转为 NumPy 数组
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    feature_maps_transform_data = feature_maps_transform_data.cpu().numpy()
    
    # 标准化特征数据
    scaler = StandardScaler()
    normalized_fit_data = scaler.fit_transform(feature_maps_fit_data.reshape(-1, feature_maps_fit_data.shape[-1]))  # (N, D)
    normalized_transform_data = scaler.transform(feature_maps_transform_data.reshape(-1, feature_maps_transform_data.shape[-1]))  # (N, D)
    
    # 运行 DBSCAN 聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(normalized_fit_data)  # 获取聚类标签
    print(f"labels.max = {labels.max()} labels.min= {labels.min()}")

    # 将标签扩展回输入形状
    labels_reshaped = labels.reshape(len(transform_experiments), -1)  # B x (H * W)
    
    for i, experiment in enumerate(transform_experiments):
        dbscan_img = labels_reshaped[i]  # (H * W)
        print(f"dbscan_img.shape = {dbscan_img.shape}")
        
        # 将 1D 聚类标签转为可视化图像
        h = w = int(sqrt(dbscan_img.shape[0]))
        dbscan_img = dbscan_img.reshape(h, w)
        
        # 将聚类标签标准化到 [0, 255]
        unique_labels = np.unique(dbscan_img)
        dbscan_img = (dbscan_img - dbscan_img.min()) / (dbscan_img.max() - dbscan_img.min())
        dbscan_img = Image.fromarray((dbscan_img * 255).astype(np.uint8))
        
        # 调整图像大小并保存
        dbscan_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(dbscan_img)
        os.makedirs(save_dir, exist_ok=True)
        dbscan_img.save(os.path.join(save_dir, f"_time_{t}_experiment_{i}_dbscan.png"))

def visualize_and_save_features_kmeans(feature_maps_fit_data, feature_maps_transform_data, 
                                       transform_experiments, t, save_dir, n_clusters=15):
    """
    对特征图数据进行 K-Means 聚类，并将结果保存为图像。
    
    参数：
        - feature_maps_fit_data: 用于 K-Means 拟合的特征图数据 (tensor or numpy array)。
        - feature_maps_transform_data: 待转换的特征图数据 (tensor or numpy array)。
        - transform_experiments: 对应的实验数据，控制批次处理 (list)。
        - t: 时间步，用于保存文件名。
        - save_dir: 保存路径。
        - n_clusters: 聚类簇的数量。
    
    输出：
        - 将聚类结果可视化为图像并保存。
    """
    # 将张量转为 NumPy 数组
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    feature_maps_transform_data = feature_maps_transform_data.cpu().numpy()
    
    # 标准化特征数据
    scaler = StandardScaler()
    normalized_fit_data = scaler.fit_transform(feature_maps_fit_data.reshape(-1, feature_maps_fit_data.shape[-1]))  # (N, D)
    normalized_transform_data = scaler.transform(feature_maps_transform_data.reshape(-1, feature_maps_transform_data.shape[-1]))  # (N, D)
    
    best_score = 999
    k = 10
    # for ks in range(2, 20):
    #     #find best k
    #     kmeans = KMeans(n_clusters=ks, random_state=42)
    #     labels = kmeans.fit_predict(normalized_fit_data)
    #     score = silhouette_score(normalized_transform_data, labels)
    #     # silhouette_scores.append(score)
    #     # print(f"Silhouette score for k={ks}: {score}")
    #     if abs(score - 1.0) < best_score:
    #         k = ks
    #         best_score = abs(score - 1.0)
    # 运行 K-Means 聚类
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(normalized_fit_data)  # 获取聚类标签
    
    # 将标签扩展回输入形状
    labels_reshaped = labels.reshape(len(transform_experiments), -1)  # B x (H * W)
    
    for i, experiment in enumerate(transform_experiments):
        kmeans_img = labels_reshaped[i]  # (H * W)
        print(f"kmeans_img.shape = {kmeans_img.shape}")
        
        # 将 1D 聚类标签转为可视化图像
        h = w = int(sqrt(kmeans_img.shape[0]))
        kmeans_img = kmeans_img.reshape(h, w)
        
        # 将聚类标签标准化到 [0, 255]
        kmeans_img = (kmeans_img - kmeans_img.min()) / (kmeans_img.max() - kmeans_img.min())
        kmeans_img = Image.fromarray((kmeans_img * 255).astype(np.uint8))
        
        # 调整图像大小并保存
        kmeans_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(kmeans_img)
        os.makedirs(save_dir, exist_ok=True)
        kmeans_img.save(os.path.join(save_dir, f"_time_{t}_experiment_{i}_kmeans.png"))

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image
