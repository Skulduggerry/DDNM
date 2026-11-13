import os
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils as tvu
import tqdm
from deeplens import GeoLens
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
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


def generate_tombo_psfs(
        lens_file,
        channels,
        kernel_size,
        psf_grid_resolution=256,
        model="geometric",
        spp=16384,
        device="cuda",
):
    assert kernel_size % 2 == 1

    # Load the optical system.
    lens = GeoLens(
        filename=lens_file,
        device=device,
        dtype=torch.float64,
    )

    lens.set_sensor_res((psf_grid_resolution, psf_grid_resolution))

    wavelengths = {
        1: [0.588],
        3: [0.656, 0.588, 0.486],
    }[channels]

    # All lenses are identical and centered.
    # Therefore, all use the same central field point.
    field_point = torch.tensor(
        [0.0, 0.0, -10000.0],
        device=device,
        dtype=torch.float64,
    )

    psfs = []

    with torch.no_grad():

        for wavelength in wavelengths:

            psf = lens.psf(
                points=field_point,
                ks=kernel_size,
                wvln=wavelength,
                model=model,
                spp=spp,
                recenter=True,
            )

            psf = psf.clamp_min(0.0)
            psf_sum = psf.sum()

            if not torch.isfinite(psf_sum) or psf_sum <= 1e-12:
                raise RuntimeError(
                    f"Invalid PSF for wavelength {wavelength}"
                )

            # Normalize the intensity kernel.
            psf = psf / psf_sum

            psfs.append(psf)

    return torch.stack(psfs).float()


def extract_fractional_patches(x, offsets, psf_radius):
    b, c, h, w = x.shape

    if h <= 1 or w <= 1:
        raise ValueError("Image dimensions must exceed 1.")

    offsets = offsets.to(device=x.device, dtype=x.dtype)

    max_offset = math.ceil(offsets.abs().max().item())
    margin = psf_radius + max_offset

    x_padded = F.pad(x, (margin, margin, margin, margin), mode="constant", value=0.0)

    hp, wp = x_padded.shape[-2:]

    yy, xx = torch.meshgrid(
        torch.arange(-psf_radius, h + psf_radius, device=x.device, dtype=x.dtype),
        torch.arange(-psf_radius, w + psf_radius, device=x.device, dtype=x.dtype),
        indexing="ij",
    )

    patches = []

    for dy, dx in offsets:
        gx = xx + dx + margin
        gy = yy + dy + margin

        gx = 2.0 * gx / (wp - 1) - 1.0
        gy = 2.0 * gy / (hp - 1) - 1.0

        grid = torch.stack([gx, gy], dim=-1)

        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

        patch = F.grid_sample(x_padded, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

        patches.append(patch)

    return torch.stack(patches, dim=1)


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


class TOMBOForwardOperator:
    def __init__(self,
                 config,
                 has_noise,
                 lens_file="./lenses/cellphone80deg.json",
                 device="cuda",
                 dtype=torch.float32,
                 model="geometric",
                 spp=16384):
        self.device = device
        self.dtype = dtype

        self.repeat_cnt = int(config.ddnm.repeat_cnt)
        self.shift = float(config.ddnm.shift)

        self.output_size = int(config.ddnm.output_size)
        self.image_size = int(config.data.image_size)

        self.channels = int(config.data.channels)
        self.kernel_size = int(config.ddnm.kernel_size)

        num_units = self.repeat_cnt ** 2

        # Output dimensions of one lens unit.
        self.lens_out_size = self.output_size // self.repeat_cnt

        assert self.lens_out_size * self.repeat_cnt == self.output_size

        # Exact integer downsampling factor.
        assert self.image_size % self.lens_out_size == 0

        self.downsample_factor = (self.image_size // self.lens_out_size)

        assert self.kernel_size % 2 == 1

        self.rtol = float(config.ddnm.cg_rtol)
        self.rho = float(config.ddnm.cg_rho) if has_noise else 0
        self.max_iter = int(config.ddnm.cg_max_iter)

        assert self.rtol >= 0
        assert self.rho >= 0
        assert self.max_iter > 0

        # =========================
        # Sampling offsets
        # =========================

        offsets = [
            (row - (self.repeat_cnt - 1) / 2, col - (self.repeat_cnt - 1) / 2,)
            for row in range(self.repeat_cnt)
            for col in range(self.repeat_cnt)
        ]
        self.offsets = torch.tensor(offsets, dtype=torch.float32, device=device) * self.shift

        # =========================
        # Generate shared PSFs
        # =========================

        psfs = generate_tombo_psfs(
            lens_file=lens_file,
            channels=self.channels,
            kernel_size=self.kernel_size,
            psf_grid_resolution=self.image_size,  # PSF is calculated at high resolution.
            model=model,
            spp=spp,
            device=device,
        )

        # Shape: [C, K, K]
        self.psfs = psfs

        assert self.psfs.shape == (
            self.channels,
            self.kernel_size,
            self.kernel_size,
        )

        assert len(offsets) == num_units

        # =========================
        # Generate pseudo inverse
        # =========================
        with torch.enable_grad():
            x_template = torch.zeros(
                (1, config.data.channels, config.data.image_size, config.data.image_size),
                device=self.device,
                dtype=self.dtype,
            )
            _, vjp_fn = torch.func.vjp(lambda x: self.forward(x), x_template)
            self.forward_transpose = lambda y: vjp_fn(y)[0]

        print("TOMBO operator initialized:")
        print("  Input size:", self.image_size)
        print("  Number of units:", num_units)
        print("  Extracted patch size:", self.image_size + 2 * (self.kernel_size // 2))
        print("  Sensor output per unit:", self.lens_out_size)
        print("  Downsampling factor:", self.downsample_factor)
        print("  PSF shape:", tuple(self.psfs.shape))
        print("  Sampling offsets:", self.offsets.cpu().tolist())

    def forward(self, x):
        b, c, h, w = x.shape

        assert c == self.channels
        assert h == self.image_size
        assert w == self.image_size

        num_units = self.repeat_cnt ** 2

        # =========================
        # Extract TOMBO sub-images
        # =========================

        patches = extract_fractional_patches(x, self.offsets)
        # [B, N, C, H, W]

        # =========================
        # Optical aberration
        # =========================

        kernel_size = self.kernel_size
        padding = kernel_size // 2

        # Expand the shared PSFs to all patches.
        #
        # [C, K, K] -> [B*N*C, 1, K, K]

        weights = (
            self.psfs
            .to(device=x.device, dtype=x.dtype)
            .unsqueeze(0).unsqueeze(0)
            .expand(b, num_units, c, kernel_size, kernel_size)
            .reshape(b * num_units * c, 1, kernel_size, kernel_size)
        )

        # Treat every patch/channel combination as
        # an independent convolution channel.

        optical_input = patches.reshape(b * num_units, c, h, w)

        weights = self.psfs.to(device=x.device, dtype=x.dtype, ).unsqueeze(1)

        optical_input = F.pad(
            optical_input,
            (padding, padding, padding, padding),
            mode="reflect",
        )

        optical_output = F.conv2d(
            optical_input,
            weights.flip((-2, -1)),
            groups=c,
        )

        patches = optical_output.reshape(b, num_units, c, h, w)

        # =========================
        # Downsampling
        # =========================

        patches = patches.reshape(b * num_units, c, h, w)

        patches = F.avg_pool2d(
            patches,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )

        assert patches.shape[-2:] == (self.lens_out_size, self.lens_out_size,)

        # =========================
        # Assembly of sub-images
        # =========================

        patches = patches.reshape(b, self.repeat_cnt, self.repeat_cnt, c, self.lens_out_size, self.lens_out_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        patches = patches.contiguous().reshape(b, c, self.output_size, self.output_size)

        return patches

    def pinv(self, y):
        b, c, h, w = y.shape

        assert b == 1, "Current CG implementation supports batch size 1."
        assert c == self.channels
        assert h == w == self.output_size

        with torch.no_grad():
            b = self.forward_transpose(y)
            AtA_rho_fn = lambda x: self.forward_transpose(self.forward(x)) + self.rho * x

            return cg(AtA_rho_fn, b, rtol=self.rtol, max_iter=self.max_iter)


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

        print('Run Simplified DDNM, without SVD.',
              f'{self.config.time_travel.T_sampling} sampling steps.',
              f'travel_length = {self.config.time_travel.travel_length},',
              f'travel_repeat = {self.config.time_travel.travel_repeat}.',
              f'Task: {self.args.deg}.'
              )
        self.simplified_ddnm_plus(model, cls_fn)

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
            tombo = TOMBOForwardOperator(config, self.args.add_noise, './lenses/cellphone80deg.json', self.device)
            A = lambda x: tombo.forward(x)
            Ap = lambda y: tombo.pinv(y)
        else:
            raise NotImplementedError("degradation type not supported")

        args.sigma_y = 2 * args.sigma_y  # to account for scaling to [-1,1]
        sigma_y = args.sigma_y

        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        pbar = tqdm.tqdm(val_loader)

        avg_mse = 0.0
        avg_psnr = 0.0
        avg_lpips = 0.0
        avg_norm_error = 0.0

        avg_mse_Apy = 0.0
        avg_psnr_Apy = 0.0
        avg_lpips_Apy = 0.0
        avg_norm_error_Apy = 0.0

        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex",
            normalize=True
        ).to(self.device)

        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A(x_orig)

            if self.args.add_noise:  # for denoising test
                y = get_gaussian_noisy_img(y, sigma_y)

            if config.sampling.batch_size != 1:
                raise ValueError("please change the config file to set batch size as 1")

            Apy = Ap(y)

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
                        xt = xs[-1].to('cuda')

                        et = model(xt, t)

                        if et.size(1) == 6:
                            et = et[:, :3]

                        # Eq. 12
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                        # Eq. 19
                        sigma_t = (1 - at_next).sqrt()
                        if sigma_t >= at_next * sigma_y:
                            lambda_t = 1.
                            gamma_t = (sigma_t ** 2 - (at_next * sigma_y) ** 2).sqrt()
                        else:
                            lambda_t = sigma_t / (at_next * sigma_y)
                            gamma_t = 0.

                        # Eq. 17
                        eta = self.args.eta
                        x0_t_hat = x0_t - lambda_t * Ap(A(x0_t) - y)
                        xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (
                                eta * torch.randn_like(x0_t) + (1 - eta ** 2) ** 0.5 * et)

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

            Apy = inverse_data_transform(config, Apy[0])[:, 4:252, 4:252].unsqueeze(0).to(self.device)
            diff_Apy = Apy - orig

            mse = torch.mean(diff ** 2)
            psnr = 10 * torch.log10(1 / mse)
            lpips = lpips_metric(orig, reconstructed)
            norm_error = torch.norm(diff) / torch.norm(orig)
            avg_mse += mse
            avg_psnr += psnr
            avg_lpips += lpips
            avg_norm_error += norm_error

            mse_Apy = torch.mean(diff_Apy ** 2)
            psnr_Apy = 10 * torch.log10(1 / mse_Apy)
            lpips_Apy = lpips_metric(orig, Apy)
            norm_error_Apy = torch.norm(diff_Apy) / torch.norm(orig)
            avg_mse_Apy += mse_Apy
            avg_psnr_Apy += psnr_Apy
            avg_lpips_Apy += lpips_Apy
            avg_norm_error_Apy += norm_error_Apy

            with open(os.path.join(self.args.image_folder, f"{idx_so_far}/errors.txt"), mode='w') as f:
                f.write(f"""
                Reconstructed:
                MSE={mse}
                PSNR={psnr}
                LPIPS={lpips}
                Norm Error={norm_error}

                Apy:
                MSE={mse_Apy}
                PSNR={psnr_Apy}
                LPIPS={lpips_Apy}
                Norm Error={norm_error_Apy}
                """)
            idx_so_far += y.shape[0]

            pbar.set_description("LPIPS: %.4f" % (avg_lpips / (idx_so_far - idx_init)))

        avg_mse /= (idx_so_far - idx_init)
        avg_psnr /= (idx_so_far - idx_init)
        avg_lpips /= (idx_so_far - idx_init)
        avg_norm_error /= (idx_so_far - idx_init)

        avg_mse_Apy /= (idx_so_far - idx_init)
        avg_psnr_Apy /= (idx_so_far - idx_init)
        avg_lpips_Apy /= (idx_so_far - idx_init)
        avg_norm_error_Apy /= (idx_so_far - idx_init)

        with open(os.path.join(self.args.image_folder, f"errors.txt"), mode='w') as f:
            f.write(f"""
            Reconstructed:
            Total Average MSE={avg_mse}
            Total Average PSNR={avg_psnr}
            Total Average LPIPS={avg_lpips}
            Total Average Norm Error={avg_norm_error}

            Apy:
            Total Average MSE={avg_mse_Apy}
            Total Average PSNR={avg_psnr_Apy}
            Total Average LPIPS={avg_lpips_Apy}
            Total Average Norm Error={avg_norm_error_Apy}
            """)

        print("Total Average PSNR: %.2f" % avg_psnr)
        print("Total Average LPIPS: %.2f" % avg_lpips)
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
