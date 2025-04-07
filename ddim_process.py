import argparse, os, sys, glob, datetime, yaml
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
# import tqdm
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

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
        default=0.0
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
    parser.add_argument(
        "--inverse_prompt",
        type=str,
        nargs="?",
        help="inverse prompt for ddim sampling",
        default=""
    )
    parser.add_argument(
        "--img",
        type=str,
        nargs="?",
        help="path to input image",
        default="input.png"
    )
    return parser

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

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model

def fetch_model_path(parser):
    opt , unknown = parser.parse_known_args()
    ckpt = None
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            model_dir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'model_dir is {model_dir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            model_dir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        model_dir = opt.resume.rstrip("/")
        ckpt = os.path.join(model_dir, "model.ckpt")

    # if opt.logdir != "none":
    #     locallog = logdir.split(os.sep)[-1]
    #     if locallog == "": locallog = logdir.split(os.sep)[-2]
    #     print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
    #     logdir = os.path.join(opt.logdir, locallog)

    return opt , model_dir , ckpt , unknown

def load_model_config(opt , model_dir , unknown):
    base_configs = sorted(glob.glob(os.path.join(model_dir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    print(config)
    return config

def load_model_by_parser(process_config,parser , gpu , eval_mode):
    opt , model_dir , ckpt , unknown = fetch_model_path(parser)
    config = load_model_config(opt , model_dir , unknown)
    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    process_config.update({"model": model, "global_step": global_step, "model_dir": model_dir , "opt": opt})
    return model, global_step , model_dir

def prepare_dirs(process_config):
    print("logging to:")
    logdir = os.path.join(process_config["opt"].logdir, "samples", f"{process_config["global_step"]:08}", process_config["now"])
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")
    print(logdir)
    process_config.update({"logdir": logdir})
    print(75 * "=")
    process_config.update({"prepare_dir_list":[imglogdir, numpylogdir]})

def gen_dirs(process_config):
    for dir_path in process_config["prepare_dir_list"]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def undo_prepare_dirs(process_config):
    for dir_path in process_config["prepare_dir_list"]:
        if os.path.exists(dir_path):
            os.removedirs(dir_path)

def log_sampling_info(process_config):
    sampling_file = os.path.join(process_config["logdir"], "sampling_config.yaml")
    sampling_conf = vars(process_config["opt"])

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

def prepare_env(process_config):
    prepare_dirs(process_config)
    gen_dirs(process_config)
    log_sampling_info(process_config)

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

def gen_prompt(model , prompt_str):
    prompt_str_new = prompt_str if prompt_str is not None else ""
    return model.get_learned_conditioning([prompt_str_new])


@torch.no_grad()
def ddim_sampling(model , img_path , batch_size , inverse_prompt , custom_steps , eta , ddim_inversion_steps = 999):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert os.path.isfile(img_path), f"Cannot find {img_path}"
    init_image = load_img(img_path).to(device)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
    sampler = DDIMSampler(model)
    c_inverse = model.get_learned_conditioning([""])
    z_enc, _ = sampler.encode_ddim(init_latent,conditioning=c_inverse , unconditional_conditioning=c_inverse, num_steps=ddim_inversion_steps)
    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]
    bs = shape[0]
    shape = shape[1:]
    c = model.get_learned_conditioning([inverse_prompt])
    samples, intermediates = sampler.sample(custom_steps, conditioning=c,batch_size=bs,callback_ddim_timesteps=None, shape=shape,log_every_t=10, eta=eta, verbose=False,x_T=z_enc,callback=None)
    x_sample = model.decode_first_stage(samples)

    return x_sample, intermediates

@torch.no_grad()
def ddim_inversion(model , img_path , inverse_prompt , device , ddim_inversion_steps = 999):
    init_image = load_img(img_path , device)
    init_latent = load_init_latent(model , init_image)
    sampler = DDIMSampler(model)
    prompt = gen_prompt(model , inverse_prompt)
    z_enc, _ = sampler.encode_ddim(init_latent,conditioning=prompt , unconditional_conditioning=inverse_prompt, num_steps=ddim_inversion_steps)
    return z_enc , sampler

@torch.no_grad()
def ddim_inversion(model , init_latent , inverse_prompt , device , start_code_path="" , ddim_inversion_steps = 999):
    sampler = DDIMSampler(model)
    prompt = gen_prompt(model , inverse_prompt)
    z_enc = None
    if os.path.exists(start_code_path):
        z_enc = torch.load(start_code_path).to(device) 
    else:
        z_enc , _= sampler.encode_ddim(init_latent,conditioning=prompt , unconditional_conditioning=inverse_prompt, num_steps=ddim_inversion_steps)
    return z_enc , sampler

@torch.no_grad()
def ddim_process_sampling(model , sampler , x_T ,
                          custom_steps,
                          conditioning,
                          batch_size , eta , verbose,
                          callback_ddim_timesteps=None , callback=None,
                          injected_features=None,
                          blend_callback=None,
                          edit_dir = None,
                          edit_scale = 1.0,
                          unconditional_conditioning=None,
                          # unconditional_conditioning=None,
                          unconditional_guidance_scale=1.,
                          noise_extract_step=None , noise_extract_callback=None):
    shape = [batch_size,
                model.model.diffusion_model.in_channels,
                model.model.diffusion_model.image_size,
                model.model.diffusion_model.image_size]
    bs = shape[0]
    shape = shape[1:]
    # print(f"got ddim callback_ddim_timesteps: {callback_ddim_timesteps}")
    #TODO: callback_ddim_timesteps doesn't mean an accurate number of steps but the total step instead
    samples, intermediate = sampler.sample(custom_steps,
                                conditioning=conditioning,
                                batch_size=bs,
                                callback_ddim_timesteps=callback_ddim_timesteps,
                                shape=shape,
                                log_every_t=1,
                                eta=eta,
                                verbose=verbose,
                                x_T=x_T,
                                callback=callback,
                                edit_scale=edit_scale,
                                edit_dir=edit_dir,
                                blend_callback=blend_callback,
                                unconditional_conditioning=unconditional_conditioning,
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                injected_features=injected_features,
                                noise_extract_time=noise_extract_step,
                                noise_extract_function=noise_extract_callback)
    
    x_sample = model.decode_first_stage(samples)
    return x_sample , intermediate

def load_init_image(img_path , device):
    assert os.path.isfile(img_path), f"Cannot find {img_path}"
    init_image = load_img(img_path).to(device)
    return init_image

def load_init_latent(model , init_image):
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
    return init_latent

def save_samples(save_path, samples, intermediates=None):
    for i , sample in enumerate(samples):
        if sample.dim() == 4:
            img = custom_to_pil(sample[0])
        else:
            img = custom_to_pil(sample)

        if save_path.endswith(".png"):
            img_path = save_path
        else:
            img_path = os.path.join(save_path, f"sample_{i}.png")
        print(f"Saving sample to {img_path}")
        img.save(img_path)

    # for i , intermediate in enumerate(intermediates):
    #     if intermediate.dim() == 4:
    #         img = custom_to_pil(intermediate[0])
    #     else:
    #         img = custom_to_pil(intermediate)
    #     img_path = os.path.join(save_path, f"intermediate_{i}.png")
    #     img.save(img_path)

    print("Samples saved!")

def ddim_process(parser):
    # ddim interface for internal calling
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # parser = get_parser()
    process_config = {}
    process_config["now"] = now
    load_model_by_parser(process_config , parser , gpu=True , eval_mode=True)
    prepare_env(process_config)
    try:
        # ddim sampling
        samples, intermediates = ddim_sampling(process_config["model"], process_config["opt"].img, process_config["opt"].batch_size, process_config["opt"].inverse_prompt, process_config["opt"].custom_steps, process_config["opt"].eta)
        save_samples(process_config["prepare_dir_list"][0], samples, intermediates)
        pass
    except Exception as e:
        print(e)
        undo_prepare_dirs(process_config)
        raise e
    print("Sampling done!")

if __name__ == "__main__":
    parser = get_parser()
    ddim_process(parser)