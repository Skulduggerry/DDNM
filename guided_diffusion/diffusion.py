import os
import random

import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.utils as tvu
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import tqdm

from deeplens import GeoLens

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.svd_ddnm import ddnm_diffusion, ddnm_plus_diffusion
from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


import torch
from deeplens import GeoLens


def generate_tombo_psfs(
        lens_file,
        field_points,
        channels=1,
        kernel_size=15,
        model="geometric",
        spp=16384,
        device="cuda",
):
    assert kernel_size % 2 == 1

    lens = GeoLens(
        filename=lens_file,
        device=device,
        dtype=torch.float64,
    )

    wavelengths = {
        1: [0.588],
        3: [0.656, 0.588, 0.486],
    }

    wavelengths = wavelengths[channels]

    psfs = []

    with torch.no_grad():

        for point in field_points:

            unit_psfs = []

            for wavelength in wavelengths:
                psf = lens.psf(
                    points=point.to(
                        device=device,
                        dtype=torch.float64,
                    ),
                    ks=kernel_size,
                    wvln=wavelength,
                    model=model,
                    spp=spp,
                    recenter=True,
                )

                # Ensure a valid normalized intensity kernel.
                psf = psf.clamp_min(0.0)
                psf = psf / psf.sum().clamp_min(1e-12)

                unit_psfs.append(psf)

            psfs.append(torch.stack(unit_psfs))

    return torch.stack(psfs).float()


psfs = None


def apply_A(x, config):
    repeat_cnt = config.ddnm.repeat_cnt
    shift = config.ddnm.shift
    out_size = config.ddnm.output_size

    b, c, h, w = x.shape
    assert h == w

    size = w - (repeat_cnt - 1) * shift
    num_units = repeat_cnt ** 2

    global psfs
    if psfs is None:
        field_points = torch.tensor(
            [
                [0.0, 0.0, -10000.0]
                for _ in range(num_units)
            ],
            dtype=torch.float64,
            device="cuda",
        )

        psfs = generate_tombo_psfs(
            lens_file="./lenses/cellphone80deg.json",
            field_points=field_points,
            channels=config.data.channels,
            kernel_size=15,
            model="geometric",
            spp=16384,
            device="cuda",
        )

    # =========================
    # Extract TOMBO sub-images
    # =========================
    patches = x.unfold(2, size, shift).unfold(3, size, shift)

    patches = (
        patches
        .permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .reshape(b * num_units, c, size, size)
    )

    # =========================
    # Optical aberration
    # =========================
    patches = patches.reshape(b, num_units, c, size, size)

    assert psfs.shape[0] == num_units
    assert psfs.shape[1] == c
    assert psfs.shape[-1] == psfs.shape[-2]

    kernel_size = psfs.shape[-1]
    padding = kernel_size // 2

    assert kernel_size % 2 == 1
    assert size > padding

    weights = (
        psfs.to(device=x.device, dtype=x.dtype)
        .unsqueeze(0)
        .expand(b, -1, -1, -1, -1)
        .reshape(b * num_units * c, 1, kernel_size, kernel_size)
    )

    # Reshape all patches into independent channels.
    optical_input = patches.reshape(
        1, b * num_units * c, size, size
    )

    # Boundary handling: reflection padding.
    optical_input = torch.nn.functional.pad(
        optical_input,
        (padding, padding, padding, padding),
        mode="reflect",
    )

    # Apply the unit-specific optical PSFs.
    # Groups ensure that no lens unit mixes with another.

    optical_output = torch.nn.functional.conv2d(
        optical_input,
        weights.flip((-2, -1)),
        groups=b * num_units * c,
    )

    # Restore original layout.
    patches = optical_output.reshape(
        b, num_units, c, size, size
    )

    # =========================
    # Downsampling
    # =========================

    patches = patches.reshape(
        b * num_units, c, size, size
    )

    lens_out_size = out_size // repeat_cnt
    assert lens_out_size * repeat_cnt == out_size

    patches = torch.nn.functional.adaptive_avg_pool2d(
        patches,
        (lens_out_size, lens_out_size),
    )

    # reshape + permute: (b * repeat_cnt ** 2, c, size, size) -> (b, c, out_size, out_size)
    patches = patches.reshape(b, repeat_cnt, repeat_cnt, c, lens_out_size, lens_out_size).permute(0, 3, 1, 4, 2, 5)
    patches = patches.contiguous().reshape(b, c, out_size, out_size)

    return patches


def cg(A_op, b, rtol, max_iter):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()

    rsold = torch.sum(r * r)
    bnorm = torch.linalg.vector_norm(b).clamp_min(1e-12)

    for i in range(max_iter):
        Ap = A_op(p)
        denom = torch.sum(p * Ap)

        if not torch.isfinite(denom) or denom <= 1e-20:
            break

        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap

        if torch.linalg.vector_norm(r) / bnorm < rtol:
            break

        rsnew = torch.sum(r * r)
        if not torch.isfinite(rsnew):
            break

        p = r + (rsnew / rsold.clamp_min(1e-30)) * p
        rsold = rsnew

    return x


def make_A_At(config, x_template):
    A_fn = lambda x: apply_A(x, config)

    with torch.enable_grad():
        _, vjp_fn = torch.func.vjp(A_fn, x_template)

    def At_fn(y):
        with torch.enable_grad():
            return vjp_fn(y)[0]

    return A_fn, At_fn


A_fn, At_fn = None, None


def apply_Ap_cg(y, current_t, config):
    global A_fn, At_fn

    b, c, h, w = y.shape
    assert h == w

    rtol = float(config.ddnm.cg_rtol)
    rho_min = float(config.ddnm.cg_rho_min)
    rho_max = float(config.ddnm.cg_rho_max)
    max_iter = int(config.ddnm.cg_max_iter)

    progress = (1000 - current_t) / 1000
    rho = rho_min + progress * (rho_max - rho_min)

    if A_fn is None or At_fn is None:
        x_template = torch.zeros(
            (1, config.data.channels, config.data.image_size, config.data.image_size),
            device=y.device,
            dtype=y.dtype,
        )
        A_fn, At_fn = make_A_At(config, x_template)

    with torch.no_grad():
        At_y = At_fn(y)
        b = At_y

        AtA_rho_fn = lambda x: At_fn(A_fn(x)) + rho * x

        return cg(AtA_rho_fn, b, rtol=rtol, max_iter=max_iter)


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, simplified):
        cls_fn = None
        if self.config.model.type == 'simple':
            model = Model(self.config)

            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            elif self.config.data.dataset == 'CelebA_HQ':
                name = 'celeba_hq'
            else:
                raise ValueError
            if name != 'celeba_hq':
                ckpt = get_ckpt_path(f"ema_{name}", prefix=self.args.exp)
                print("Loading checkpoint {}".format(ckpt))
            elif name == 'celeba_hq':
                ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
                if not os.path.exists(ckpt):
                    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',
                             ckpt)
            else:
                raise ValueError
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        elif self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                    self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                            self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                        ckpt)

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_classifier.pt' % (
                    self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    image_size = self.config.data.image_size
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_classifier.pt' % image_size,
                        ckpt)
                classifier = create_classifier(**args_to_dict(self.config.classifier, classifier_defaults().keys()))
                classifier.load_state_dict(torch.load(ckpt, map_location=self.device))
                classifier.to(self.device)
                if self.config.classifier.classifier_use_fp16:
                    classifier.convert_to_fp16()
                classifier.eval()
                classifier = torch.nn.DataParallel(classifier)

                import torch.nn.functional as F
                def cond_fn(x, t, y):
                    with torch.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        return torch.autograd.grad(selected.sum(), x_in)[0] * self.config.classifier.classifier_scale

                cls_fn = cond_fn

        if simplified:
            print('Run Simplified DDNM, without SVD.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                  )
            self.simplified_ddnm_plus(model, cls_fn)
        else:
            print('Run SVD-based DDNM.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                  )
            self.svd_based_ddnm_plus(model, cls_fn)

    def simplified_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation operator
        print("args.deg:", args.deg)
        if args.deg == 'diy_cg':
            A = lambda x: apply_A(x, config)
            Ap = lambda y, current_t=1000: apply_Ap_cg(y=y, current_t=current_t, config=config)
        else:
            raise NotImplementedError("degradation type not supported")

        args.sigma_y = 2 * args.sigma_y  # to account for scaling to [-1,1]
        sigma_y = args.sigma_y

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_mse = 0.0
        avg_psnr = 0.0
        avg_lpips = 0.0
        avg_norm_error = 0.0
        pbar = tqdm.tqdm(val_loader)

        for x_orig, classes in pbar:
            # make x_orig a 64 bit float for higher to validate
            x_orig = x_orig.to(self.device)  # x_orig.to(self.device, dtype=torch.float64)
            x_orig = data_transform(self.config, x_orig)

            y = A(x_orig)

            if self.args.add_noise:  # for denoising test
                y = get_gaussian_noisy_img(y, sigma_y)

            if config.sampling.batch_size != 1:
                raise ValueError("please change the config file to set batch size as 1")

            Apy = Ap(y)

            # validate the pseudo inverse with high precision
            # ApAApy = Ap(A(Apy))
            # error = torch.norm(ApAApy - Apy) / torch.norm(Apy)
            # print("Error between Ap(y) and Ap(A(Ap(y))): %.16f" % error)
            # assert (error < 1e-10)

            # make y a float32 to match the network
            # y = y.to(dtype=torch.float32)

            os.makedirs(os.path.join(self.args.image_folder, f"{idx_so_far}"), exist_ok=True)
            for i in range(len(Apy)):
                tvu.save_image(
                    inverse_data_transform(config, y[i]),
                    os.path.join(self.args.image_folder, f"{idx_so_far}/y.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, Apy[i]),
                    os.path.join(self.args.image_folder, f"{idx_so_far}/Apy.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"{idx_so_far}/orig.png")
                )

            # init x_T
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]

                times = get_schedule_jump(config.time_travel.T_sampling,
                                          config.time_travel.travel_length,
                                          config.time_travel.travel_repeat,
                                          )
                time_pairs = list(zip(times[:-1], times[1:]))

                # reverse diffusion sampling
                for i, j in tqdm.tqdm(time_pairs):
                    i, j = i * skip, j * skip
                    if j < 0: j = -1

                    if j < i:  # normal sampling
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        sigma_t = (1 - at_next).sqrt()
                        xt = xs[-1].to('cuda')

                        et = model(xt, t)

                        if et.size(1) == 6:
                            et = et[:, :3]

                        # Eq. 12
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                        # Eq. 19
                        if sigma_t >= at_next * sigma_y:
                            lambda_t = 1.
                            gamma_t = (sigma_t ** 2 - (at_next * sigma_y) ** 2).sqrt()
                        else:
                            lambda_t = sigma_t / (at_next * sigma_y)
                            gamma_t = 0.

                        # Eq. 17
                        x0_t_hat = x0_t - lambda_t * Ap(y=A(x0_t) - y, current_t=i)
                        eta = self.args.eta

                        c1 = (1 - at_next).sqrt() * eta
                        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                        # different from the paper, we use DDIM here instead of DDPM
                        xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                        x0_preds.append(x0_t.to('cpu'))
                        xs.append(xt_next.to('cpu'))

                        if i % 100 == 0:
                            os.makedirs(f"{self.args.image_folder}/debug_ddnm/x0_t/", exist_ok=True)
                            tvu.save_image(
                                inverse_data_transform(config, x0_t[0]),
                                f"{self.args.image_folder}/debug_ddnm/x0_t/x0_t_{i:04d}.png"
                            )

                            os.makedirs(f"{self.args.image_folder}/debug_ddnm/x0_t_hat/", exist_ok=True)
                            tvu.save_image(
                                inverse_data_transform(config, x0_t_hat[0]),
                                f"{self.args.image_folder}/debug_ddnm/x0_t_hat/x0_t_hat_{i:04d}.png"
                            )

                            os.makedirs(f"{self.args.image_folder}/debug_ddnm/xt_next/", exist_ok=True)
                            tvu.save_image(
                                inverse_data_transform(config, xt_next[0]),
                                f"{self.args.image_folder}/debug_ddnm/xt_next/xt_next_{i:04d}.png"
                            )
                    else:  # time-travel back
                        next_t = (torch.ones(n) * j).to(x.device)
                        at_next = compute_alpha(self.betas, next_t.long())
                        x0_t = x0_preds[-1].to('cuda')

                        xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                        xs.append(xt_next.to('cpu'))

                x = xs[-1]

            x = [inverse_data_transform(config, xi) for xi in x]

            tvu.save_image(
                x[0], os.path.join(self.args.image_folder, f"{idx_so_far}/ddnm.png")
            )

            orig = inverse_data_transform(config, x_orig[0])[:, 4:252, 4:252].unsqueeze(0).to(self.device)
            reconstructed = x[0][:, 4:252, 4:252].unsqueeze(0).to(self.device)
            diff = reconstructed - orig

            mse = torch.mean(diff ** 2)
            avg_mse += mse
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(self.device)
            lpips = lpips_metric(orig, reconstructed)
            avg_lpips += lpips

            norm_error = torch.norm(diff) / torch.norm(orig)
            avg_norm_error += norm_error

            with open(os.path.join(self.args.image_folder, f"{idx_so_far}/errors.txt"), mode='w') as f:
                f.write(f"""
                    MSE={mse}
                    PSNR={psnr}
                    LPIPS={lpips}
                    Norm Error={norm_error}
                """)
            idx_so_far += y.shape[0]

            print("MSE: %f" % mse)
            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_mse /= (idx_so_far - idx_init)
        avg_psnr /= (idx_so_far - idx_init)
        avg_lpips /= (idx_so_far - idx_init)
        avg_norm_error /= (idx_so_far - idx_init)

        with open(os.path.join(self.args.image_folder, f"errors.txt"), mode='w') as f:
            f.write(f"""
                Total Average MSE={avg_mse}
                Total Average PSNR={avg_psnr}
                Total Average LPIPS={avg_lpips}
                Total Average Norm Error={avg_norm_error}
            """)

        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Total Average LPIPS: %.2f" % avg_lpips)
        print("Number of samples: %d" % (idx_so_far - idx_init))

    def svd_based_ddnm_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation matrix
        deg = args.deg
        A_funcs = None
        if deg == 'cs_walshhadamard':
            compress_by = round(1 / args.deg_scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        elif deg == 'cs_blockbased':
            cs_ratio = args.deg_scale
            from functions.svd_operators import CS
            A_funcs = CS(config.data.channels, self.config.data.image_size, cs_ratio, self.device)
        elif deg == 'inpainting':
            from functions.svd_operators import Inpainting
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            A_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)
        elif deg == 'denoising':
            from functions.svd_operators import Denoising
            A_funcs = Denoising(config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'colorization':
            from functions.svd_operators import Colorization
            A_funcs = Colorization(config.data.image_size, self.device)
        elif deg == 'sr_averagepooling':
            blur_by = int(args.deg_scale)
            from functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'sr_bicubic':
            factor = int(args.deg_scale)
            from functions.svd_operators import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0

            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            A_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride=factor)
        elif deg == 'deblur_uni':
            from functions.svd_operators import Deblurring
            A_funcs = Deblurring(torch.Tensor([1 / 9] * 9).to(self.device), config.data.channels,
                                 self.config.data.image_size, self.device)
        elif deg == 'deblur_gauss':
            from functions.svd_operators import Deblurring
            sigma = 10
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.device)
            A_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, self.config.data.image_size, self.device)
        elif deg == 'deblur_aniso':
            from functions.svd_operators import Deblurring2D
            sigma = 20
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            sigma = 1
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(
                self.device)
            A_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels,
                                   self.config.data.image_size, self.device)
        elif deg == 'tombo':
            from functions.svd_operators import TomboSVD
            A_funcs = TomboSVD(channels=config.data.channels,
                               input_size=config.data.image_size,
                               output_size=config.ddnm.output_size,
                               repeat_cnt=config.ddnm.repeat_cnt,
                               shift=config.ddnm.shift,
                               blur_ksize=config.ddnm.blur_ksize,
                               blur_strength=config.ddnm.blur_strength,
                               aberration_factor=config.ddnm.aberration_factor,
                               vignette_falloff=config.ddnm.vignette_falloff,
                               mu=float(config.ddnm.mu),
                               device=self.device)
        else:
            raise ValueError("degradation type not supported")
        args.sigma_y = 2 * args.sigma_y  # to account for scaling to [-1,1]
        sigma_y = args.sigma_y

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr_tombo = 0.0
        # avg_psnr_sr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A_funcs.A(x_orig)

            b, hwc = y.size()
            hw = hwc / 3
            h = w = int(hw ** 0.5)
            y = y.reshape((b, 3, h, w))

            if self.args.add_noise:  # for denoising test
                y = get_gaussian_noisy_img(y, sigma_y)

            # cut out
            # y_sr_only = y[:, :, :64, :64]
            # y_sr_only = y_sr_only.contiguous().reshape((b, -1))
            # y = y.reshape((b, hwc))

            # from functions.svd_operators import SuperResolution
            # SR_funcs = SuperResolution(config.data.channels, config.data.image_size, 4, self.device)

            Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                         self.config.data.image_size)
            # Apy_sr_only = SR_funcs.A_pinv(y_sr_only).view(y_sr_only.shape[0],
            #                                               config.data.channels,
            #                                               self.config.data.image_size,
            #                                               self.config.data.image_size)

            os.makedirs(os.path.join(self.args.image_folder, f"{idx_so_far}"), exist_ok=True)
            for i in range(len(Apy)):
                tvu.save_image(
                    inverse_data_transform(config, y[i].reshape(self.config.data.channels, self.config.data.image_size,
                                                                self.config.data.image_size)),
                    os.path.join(self.args.image_folder, f"{idx_so_far}/y.png")
                )
                # tvu.save_image(
                #     inverse_data_transform(config, y_sr_only.reshape(self.config.data.channels, 64, 64)),
                #     os.path.join(self.args.image_folder, f"{idx_so_far}/y_sr_only.png")
                # )
                tvu.save_image(
                    inverse_data_transform(config, Apy[i]),
                    os.path.join(self.args.image_folder, f"{idx_so_far}/Apy.png")
                )
                # tvu.save_image(
                #     inverse_data_transform(config, Apy_sr_only[i]),
                #     os.path.join(self.args.image_folder, f"{idx_so_far}/Apy_sr_only.png")
                # )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"{idx_so_far}/orig.png")
                )

            # Start DDIM
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            # x_sr_only = torch.randn(
            #     y_sr_only.shape[0],
            #     config.data.channels,
            #     config.data.image_size,
            #     config.data.image_size,
            #     device=self.device,
            # )

            with torch.no_grad():
                if sigma_y == 0.:  # noise-free case, turn to ddnm
                    x, _ = ddnm_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, cls_fn=cls_fn,
                                          classes=classes, config=config)
                    # x_sr_only, _ = ddnm_diffusion(x_sr_only, model, self.betas, self.args.eta, SR_funcs, y_sr_only,
                    #                               cls_fn=cls_fn,
                    #                               classes=classes, config=config)
                else:  # noisy case, turn to ddnm+
                    x, _ = ddnm_plus_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, sigma_y, cls_fn=cls_fn,
                                               classes=classes, config=config)
                    # x_sr_only, _ = ddnm_plus_diffusion(x_sr_only, model, self.betas, self.args.eta, SR_funcs, y_sr_only,
                    #                                    sigma_y, cls_fn=cls_fn,
                    #                                    classes=classes, config=config)

            x = [inverse_data_transform(config, xi) for xi in x]
            # x_sr_only = [inverse_data_transform(config, xi) for xi in x_sr_only]

            tvu.save_image(
                x[0], os.path.join(self.args.image_folder, f"{idx_so_far}/tombo_ddnm+.png")
            )
            # tvu.save_image(
            #     x_sr_only[0], os.path.join(self.args.image_folder, f"{idx_so_far}/sr_ddnm+.png")
            # )

            orig = inverse_data_transform(config, x_orig[0])[:, 4:252, 4:252]
            diff_tombo = x[0][:, :, 4:252, 4:252].to(self.device) - orig
            # diff_sr = x_sr_only[0][:, :, 4:252, 4:252].to(self.device) - orig

            mse_tombo = torch.mean(diff_tombo ** 2)
            psnr_tombo = 10 * torch.log10(1 / mse_tombo)
            avg_psnr_tombo += psnr_tombo

            # mse_sr = torch.mean(diff_sr ** 2)
            # psnr_sr = 10 * torch.log10(1 / mse_sr)
            # avg_psnr_sr += psnr_sr

            norm_error_tombo = torch.norm(diff_tombo) / torch.norm(orig)
            # norm_error_sr = torch.norm(diff_sr) / torch.norm(orig)

            with open(os.path.join(self.args.image_folder, f"{idx_so_far}/errors.txt"), mode='w') as f:
                f.write(f"""
                    TOMBO:
                    MSE={mse_tombo}
                    PSNR={psnr_tombo}
                    Norm Error={norm_error_tombo}
                    """)
                # f.write(f"""
                #     TOMBO:
                #     MSE={mse_tombo}
                #     PSNR={psnr_tombo}
                #     Norm Error={norm_error_tombo}
                #
                #     SR:
                #     MSE={mse_sr}
                #     PSNR={psnr_sr}
                #     Norm Error={norm_error_sr}
                # """)
            idx_so_far += y.shape[0]

            # pbar.set_description(f"PSNR TOMBO={avg_psnr_tombo / (idx_so_far - idx_init):.2f}, SR={avg_psnr_sr / (idx_so_far - idx_init):.2f}")
            pbar.set_description(f"PSNR TOMBO={avg_psnr_tombo / (idx_so_far - idx_init):.2f}")

        avg_psnr_tombo = avg_psnr_tombo / (idx_so_far - idx_init)
        # avg_psnr_sr = avg_psnr_sr / (idx_so_far - idx_init)

        with open(os.path.join(self.args.image_folder, f"errors.txt"), mode='w') as f:
            f.write(f"Total Average PSNR (TOMBO)={avg_psnr_tombo}")
            # f.write(f"""
            #     Total Average PSNR (TOMBO)={avg_psnr_tombo}
            #     Total Average PSNR (SR)={avg_psnr_sr}
            # """)

        print("Total Average PSNR (TOMBO): %.2f" % avg_psnr_tombo)
        # print("Total Average PSNR (SR): %.2f" % avg_psnr_sr)
        print("Number of samples: %d" % (idx_so_far - idx_init))


# Code form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t - 1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts


def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
