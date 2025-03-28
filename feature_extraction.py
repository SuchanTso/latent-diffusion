import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import tqdm

rescale = lambda x: (x + 1.) / 2.

def load_img(path):
    image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0 , xT = None , call_back=None,
                    call_back_timesteps = None
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    c = model.get_learned_conditioning([""])# text prompt @suchan
    samples, intermediates = ddim.sample(steps, conditioning=c,batch_size=bs,callback_ddim_timesteps=call_back_timesteps, shape=shape,log_every_t=10, eta=eta, verbose=False,x_T=xT,callback=call_back)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,call_back = None , call_back_timesteps = None):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            print(f"make ddim process")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            file_path = "/data/zsc/feature_decompose_diffusion/datasets/flowers/2521408074_e6f86daf21_n.jpg"
            assert os.path.isfile(file_path)
            init_image = load_img(file_path).to(device)
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
            ddim_inversion_steps = 999
            sampler = DDIMSampler(model)
            c = model.get_learned_conditioning([""])
            z_enc, _ = sampler.encode_ddim(init_latent,conditioning=c , unconditional_conditioning=c, num_steps=ddim_inversion_steps)


            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta , xT=z_enc , call_back=call_back , call_back_timesteps = call_back_timesteps)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)
    intermediates_imgs = []
    for x_inter in intermediates['x_inter']:
        decode_img = model.decode_first_stage(x_inter)
        intermediates_imgs.append(decode_img)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    log['intermediates'] = intermediates_imgs
    print(f'Throughput for this batch: {log["throughput"]}')
    return log



def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None , save_all_features = False):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    unet_model = model.model.diffusion_model
    feature_maps_path = os.path.join(logdir, "feature_maps")
    save_feature_timesteps = custom_steps
    callback_timesteps_to_save = save_feature_timesteps
    os.makedirs(feature_maps_path, exist_ok=True)


    def ddim_sampler_callback(i):
        save_feature_maps_callback(i)

    def save_feature_maps_callback(i):
        if save_all_features:
            save_feature_maps(unet_model.input_blocks, i, "input_block")
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block in blocks:
            if not save_all_features and block_idx < 4:
                block_idx += 1
                continue
            if "ResBlock" in str(type(block[0])):
                if save_all_features or block_idx == 4:
                    save_feature_map(block[0].in_layers_features, f"{feature_type}_{block_idx}_in_layers_features_time_{i}")
                    save_feature_map(block[0].out_layers_features, f"{feature_type}_{block_idx}_out_layers_features_time_{i}")
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                save_feature_map(block[1].transformer_blocks[0].attn1.k, f"{feature_type}_{block_idx}_self_attn_k_time_{i}")
                save_feature_map(block[1].transformer_blocks[0].attn1.q, f"{feature_type}_{block_idx}_self_attn_q_time_{i}")
                save_feature_map(block[1].transformer_blocks[0].attn1.attn_per_head, f"{feature_type}_{block_idx}_self_attn_per_head_time_{i}")
                save_feature_map(block[1].transformer_blocks[0].attn1.heads, f"{feature_type}_{block_idx}_self_attn_head_time_{i}")

            block_idx += 1

    def save_feature_map(feature_map, filename):
        save_path = os.path.join(feature_maps_path, f"{filename}.pt")
        torch.save(feature_map, save_path)

    # path = logdir
    if model.cond_stage_model is not None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        # for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
        logs = make_convolutional_sample(model, batch_size=batch_size,
                                            vanilla=vanilla, custom_steps=custom_steps,
                                            eta=eta,
                                            call_back=ddim_sampler_callback,
                                            call_back_timesteps = callback_timesteps_to_save
                                            )
        n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
        save_logs(logs, logdir, n_saved=n_saved, key="intermediates")
        all_images.extend([custom_to_np(logs["sample"])])
        if n_saved >= n_samples:
            print(f'Finish after generating {n_saved} samples')
                # break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    if x.dim() == 4 :
                        img = custom_to_pil(x[0])
                    else :
                        img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved

def block_enumerate(blocks):
    block_idx = 0
    for block in blocks:
            # if block_idx < 4:
            #     block_idx += 1
            #     continue
            if "ResBlock" in str(type(block[0])):
                if block_idx == 4:
                    print(type(block[0]))
                    # save_feature_map(block[0].in_layers_features, f"{feature_type}_{block_idx}_in_layers_features_time_{i}")
                    # save_feature_map(block[0].out_layers_features, f"{feature_type}_{block_idx}_out_layers_features_time_{i}")
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                print(f"block = {block} , block_idx = {block_idx}")
                # save_feature_map(block[1].transformer_blocks[0].attn1.k, f"{feature_type}_{block_idx}_self_attn_k_time_{i}")
                # save_feature_map(block[1].transformer_blocks[0].attn1.q, f"{feature_type}_{block_idx}_self_attn_q_time_{i}")
            block_idx += 1


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu" ,weights_only=False)
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step

def modify_visualise_config(change_dir_value = None , unique_experiment_name = None):
    setup_config = OmegaConf.load("./configs/setup.yaml")
    visualise_config_file = setup_config.config.visualise_config_file
    if not os.path.exists(visualise_config_file):
        raise ValueError("Cannot find {}".format(visualise_config_file))
    if os.path.isfile(visualise_config_file):
        img_dir = os.path.join(change_dir_value, "img")
        visual_config = OmegaConf.load(visualise_config_file)
        OmegaConf.update(visual_config, "config.experiments_info_dir", [change_dir_value] , merge=False)
        OmegaConf.update(visual_config, "config.experiments_fit", [img_dir] , merge=False)
        OmegaConf.update(visual_config, "config.experiments_transform", [img_dir] , merge=False)
        if unique_experiment_name is not None:
            experiment_name = visual_config.config.experiment_name
            parts = experiment_name.split('_', 1)
            assert len(parts) >= 1
            OmegaConf.update(visual_config, "config.experiment_name", f"{parts[0]}_{unique_experiment_name}" , merge=False)
        with open(visualise_config_file, 'w') as f:
            OmegaConf.save(visual_config, f)

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    # print(model.model.diffusion_model)

    # block_enumerate(model.model.diffusion_model.output_blocks)
    try:
        run(model, imglogdir, eta=opt.eta,
            vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
            batch_size=opt.batch_size, nplog=numpylogdir , save_all_features = True)
        modify_visualise_config(change_dir_value=logdir , unique_experiment_name=now)

    except Exception as e:
        print(e)
        os.rmdir(imglogdir)
        os.rmdir(numpylogdir)
        raise e
    print("done.")


