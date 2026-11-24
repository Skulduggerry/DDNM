import math
import os
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.utils as tvu
import tqdm
from deeplens import GeoLens
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.io import read_image

from datasets import data_transform, get_dataset, inverse_data_transform
from functions.ckpt_util import download, get_ckpt_path
from guided_diffusion.models import Model
from guided_diffusion.script_util import args_to_dict, classifier_defaults, create_classifier, create_model

DEFAULT_LENS_FILE = "./lenses/edmund_65_254_tombo_lenslet.json"
VALID_MODES = {"simulate", "forward", "reverse"}


# -----------------------------------------------------------------------------
# Diffusion and image utilities
# -----------------------------------------------------------------------------


def get_gaussian_noisy_img(img: torch.Tensor, noise_level: float) -> torch.Tensor:
    return img + torch.randn_like(img) * noise_level


def get_beta_schedule(beta_schedule: str, *, beta_start: float, beta_end: float,
                      num_diffusion_timesteps: int, ) -> np.ndarray:
    """Create the beta schedule used by the diffusion process."""

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (np.exp(-x) + 1.0)

    if beta_schedule == "quad":
        betas = np.linspace(
            beta_start ** 0.5,
            beta_end ** 0.5,
            num_diffusion_timesteps,
            dtype=np.float64,
        ) ** 2
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start,
            beta_end,
            num_diffusion_timesteps,
            dtype=np.float64,
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps,
            1,
            num_diffusion_timesteps,
            dtype=np.float64,
        )
    elif beta_schedule == "sigmoid":
        steps = np.linspace(-6.0, 6.0, num_diffusion_timesteps)
        betas = sigmoid(steps) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")

    if betas.shape != (num_diffusion_timesteps,):
        raise RuntimeError("Generated beta schedule has an unexpected shape.")
    return betas


def compute_alpha(beta: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """Compute cumulative alpha values for a batch of diffusion timesteps."""
    beta_with_zero = torch.cat([beta.new_zeros(1), beta], dim=0)
    alpha_bar = (1.0 - beta_with_zero).cumprod(dim=0)
    return alpha_bar.index_select(0, timestep + 1).view(-1, 1, 1, 1)


# -----------------------------------------------------------------------------
# TOMBO optical model
# -----------------------------------------------------------------------------


def generate_tombo_psfs(
        lens_file: str,
        channels: int,
        kernel_size: int,
        target_distance_mm: float,
        psf_grid_resolution: int = 256,
        model: str = "coherent",
        spp: int | None = None,
        device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Generate normalized intensity PSFs for a target distance from the sensor.

    ``target_distance_mm`` is measured from the image/sensor plane, matching
    the z convention in the TOMBO bench plan. GeoLens points are specified
    relative to the first (object-side) surface, so the lens-to-sensor distance
    is subtracted before constructing the object point.
    """
    if kernel_size <= 0 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer.")

    wavelengths_by_channel = {
        1: [0.5876],
        3: [0.6563, 0.5876, 0.4861],
    }
    if channels not in wavelengths_by_channel:
        raise ValueError(f"Unsupported channel count for PSF generation: {channels}")

    lens = GeoLens(
        filename=lens_file,
        device=device,
        dtype=torch.float64,
    )
    lens.set_sensor_res((psf_grid_resolution, psf_grid_resolution))
    # lens.draw_layout(filename="./exp/image_samples/lens_layout.png", depth=float("inf"))

    # The bench-plan target distance z is measured from the image plane.
    # GeoLens coordinates start at the first (object-side) surface, so convert
    # the sensor-referenced distance to a first-surface-referenced point.
    target_distance_mm = float(target_distance_mm)
    if not math.isfinite(target_distance_mm) or target_distance_mm <= 0.0:
        raise ValueError("target_distance_mm must be a finite positive distance.")

    lens_to_sensor_mm = float(torch.as_tensor(lens.d_sensor).detach().cpu())
    object_distance_from_first_surface_mm = target_distance_mm - lens_to_sensor_mm
    if object_distance_from_first_surface_mm <= 0.0:
        raise ValueError(f"target_distance_mm must exceed the lens-to-sensor distance ({lens_to_sensor_mm:.6f} mm).")

    # Current TOMBO setup assumes identical, centered lens units.
    field_point = torch.tensor(
        [0.0, 0.0, -object_distance_from_first_surface_mm],
        device=device,
        dtype=torch.float64,
    )

    psfs = []
    with torch.no_grad():
        for wavelength in wavelengths_by_channel[channels]:
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
                raise RuntimeError(f"Invalid PSF for wavelength {wavelength}.")

            psfs.append(psf / psf_sum)

    return torch.stack(psfs).float()


def extract_fractional_patches(x: torch.Tensor, sampling_grid: torch.Tensor) -> torch.Tensor:
    """Extract all fractionally shifted lens patches in one grid_sample call.

    Args:
        x: Input tensor with shape [B, C, H, W].
        sampling_grid: Sampling grids with shape [N, H_out, W_out, 2].

    Returns:
        Tensor with shape [B, N, C, H_out, W_out].
    """
    batch_size, channels, height, width = x.shape
    num_units, patch_h, patch_w, _ = sampling_grid.shape

    # Expand each input over the lens dimension, then flatten B and N.
    lens_inputs = (
        x.unsqueeze(1)
        .expand(-1, num_units, -1, -1, -1)
        .reshape(batch_size * num_units, channels, height, width)
    )
    lens_grids = (
        sampling_grid.unsqueeze(0)
        .expand(batch_size, -1, -1, -1, -1)
        .reshape(batch_size * num_units, patch_h, patch_w, 2)
    )

    patches = F.grid_sample(
        lens_inputs,
        lens_grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return patches.reshape(batch_size, num_units, channels, patch_h, patch_w)


def cg(
        operator: Callable[[torch.Tensor], torch.Tensor],
        rhs: torch.Tensor,
        rtol: float,
        max_iter: int,
) -> torch.Tensor:
    """Solve a symmetric positive-semidefinite linear system with CG."""
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()

    residual_sq = torch.sum(residual * residual)
    rhs_norm = torch.linalg.vector_norm(rhs).clamp_min(1e-12)

    for _ in range(max_iter):
        operator_direction = operator(direction)
        denominator = torch.sum(direction * operator_direction)

        if not torch.isfinite(denominator) or denominator <= 1e-20:
            break

        step_size = residual_sq / denominator
        solution = solution + step_size * direction
        residual = residual - step_size * operator_direction

        if torch.linalg.vector_norm(residual) / rhs_norm < rtol:
            break

        new_residual_sq = torch.sum(residual * residual)
        if not torch.isfinite(new_residual_sq):
            break

        direction = residual + (new_residual_sq / residual_sq.clamp_min(1e-30)) * direction
        residual_sq = new_residual_sq

    return solution


class TOMBOForwardOperator:
    """Linear TOMBO degradation operator and CG-based regularized inverse.

    The current implementation uses one shared PSF per color channel for all
    lens units. The lens dimension is retained explicitly in the patch and
    convolution tensors so per-lens PSFs can be introduced later.
    """

    def __init__(
            self,
            config,
            has_noise: bool,
            convolution_method,
            lens_file: str = DEFAULT_LENS_FILE,
            device: torch.device | str = "cuda",
            dtype: torch.dtype = torch.float32,
            model: str = "coherent",
            spp: int | None = None,
            build_pinv: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype

        self.repeat_cnt = int(config.ddnm.repeat_cnt)
        self.output_size = int(config.ddnm.output_size)
        self.image_size = int(config.data.image_size)
        self.channels = int(config.data.channels)
        self.kernel_size = int(config.ddnm.kernel_size)

        self.cg_rtol = float(config.ddnm.cg_rtol)
        self.cg_rho = float(config.ddnm.cg_rho) if has_noise else 0.0
        self.cg_max_iter = int(config.ddnm.cg_max_iter)

        self.convolution_method = convolution_method
        if self.convolution_method != 'fft' and self.convolution_method != 'conv2d':
            raise ValueError(f"Invalid convolution method '{convolution_method}'. Use 'fft' or 'conv2d'")

        self._validate_config()
        self.num_units = self.repeat_cnt ** 2
        self.lens_out_size = self.output_size // self.repeat_cnt
        self.downsample_factor = self.image_size // self.lens_out_size

        # Bench-plan geometry: delta = 1024/z on low resolution pixels
        # Shift is clamped to an int! This is a problem if we have a shift that is not multiple of 1/4
        # TODO: fix this
        self.target_distance_mm = float(config.ddnm.target_distance_mm)
        self.low_resolution_disparity_px = 1024.0 / self.target_distance_mm
        self.psf_radius = int(self.kernel_size) // 2
        self.shift = int(self.low_resolution_disparity_px * self.downsample_factor)

        self.offsets = self._make_offsets()
        self.sampling_grid = self._make_sampling_grid().to(device=device, dtype=dtype)

        # Current optical model: identical centered lens units, one PSF per color.
        self.psfs = generate_tombo_psfs(
            lens_file=lens_file,
            channels=self.channels,
            kernel_size=self.kernel_size,
            psf_grid_resolution=self.image_size,
            model=model,
            spp=spp,
            device=self.device,
            target_distance_mm=self.target_distance_mm,
        ).to(device=device, dtype=dtype)
        expected_psf_shape = (
            self.channels,
            self.kernel_size,
            self.kernel_size,
        )
        if tuple(self.psfs.shape) != expected_psf_shape:
            raise RuntimeError(f"Unexpected PSF shape {tuple(self.psfs.shape)}; expected {expected_psf_shape}.")

        self.forward_transpose = None
        if build_pinv:
            self._build_forward_transpose()

        self._print_summary()

    def _validate_config(self) -> None:
        if self.repeat_cnt <= 0:
            raise ValueError("repeat_cnt must be positive.")
        if self.kernel_size <= 0 or self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer.")
        if self.output_size % self.repeat_cnt != 0:
            raise ValueError("output_size must be divisible by repeat_cnt.")
        if self.cg_rtol < 0 or self.cg_rho < 0 or self.cg_max_iter <= 0:
            raise ValueError("Invalid setting for CG algorithm.")

        if self.output_size <= 0 or self.image_size <= 0:
            raise ValueError("image_size and output_size must be positive.")
        lens_out_size = self.output_size // self.repeat_cnt
        if lens_out_size <= 0:
            raise ValueError("output_size must be at least repeat_cnt.")
        if self.image_size % lens_out_size != 0:
            raise ValueError("image_size must be divisible by output_size / repeat_cnt.")
        if self.channels not in (1, 3):
            raise ValueError("The current TOMBO PSF generator supports 1 or 3 channels.")

    def _make_offsets(self) -> torch.Tensor:
        offsets = [(row - (self.repeat_cnt - 1) / 2.0, col - (self.repeat_cnt - 1) / 2.0)
                   for row in range(self.repeat_cnt)
                   for col in range(self.repeat_cnt)]
        return torch.tensor(
            offsets,
            dtype=torch.float32,
            device=self.device,
        ) * self.shift

    def _make_sampling_grid(self) -> torch.Tensor:
        """Precompute the enlarged fractional sampling grid for each lens."""
        p = self.psf_radius
        size = self.image_size

        yy, xx = torch.meshgrid(
            torch.arange(-p, size + p, dtype=torch.float32, device=self.device),
            torch.arange(-p, size + p, dtype=torch.float32, device=self.device),
            indexing="ij",
        )

        # Shift in original-image pixel coordinates.
        sample_x = xx.unsqueeze(0) + self.offsets[:, 1, None, None]
        sample_y = yy.unsqueeze(0) + self.offsets[:, 0, None, None]

        # Normalize relative to the ACTUAL input dimensions.
        grid_x = 2.0 * sample_x / (size - 1) - 1.0
        grid_y = 2.0 * sample_y / (size - 1) - 1.0

        return torch.stack([grid_x, grid_y], dim=-1)

    def _build_forward_transpose(self) -> None:
        """Build the VJP of the linear forward operator, i.e. its transpose."""
        template = torch.zeros(
            1,
            self.channels,
            self.image_size,
            self.image_size,
            device=self.device,
            dtype=self.dtype,
        )
        with torch.enable_grad():
            _, vjp_fn = torch.func.vjp(self.forward, template)
        self.forward_transpose = lambda y: vjp_fn(y)[0]

    def _print_summary(self) -> None:
        print("TOMBO operator initialized:")
        print(f"  Input size: {self.image_size}")
        print(f"  Number of units: {self.num_units}")
        print(f"  Sensor output per unit: {self.lens_out_size}")
        print(f"  Downsampling factor: {self.downsample_factor}")
        print(f"  Bench disparity: "
              f"{self.low_resolution_disparity_px:.4f} LR px/unit, "
              f"{self.shift:.4f} HR px/unit "
              )
        print(f"  Target distance: {self.target_distance_mm:.3f} mm from image plane")
        print(f"  PSF shape: {tuple(self.psfs.shape)}")
        print(f"  Sampling offsets: {self.offsets.cpu().tolist()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fractional shifts, PSF blur, sensor integration, and mosaicing."""
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape {tuple(x.shape)}.")

        batch_size, channels, height, width = x.shape
        expected_shape = (self.channels, self.image_size, self.image_size)
        if (channels, height, width) != expected_shape:
            raise ValueError(f"Expected input shape [B, {expected_shape}], got {tuple(x.shape)}.")

        patches = extract_fractional_patches(x, self.sampling_grid)

        patch_h = height + 2 * self.psf_radius
        patch_w = width + 2 * self.psf_radius
        patches = patches.reshape(
            batch_size * self.num_units,
            channels,
            patch_h,
            patch_w,
        )

        if convolution_method == 'fft':
            fft_h = 1 << (patch_h + self.kernel_size - 2).bit_length()
            fft_w = 1 << (patch_w + self.kernel_size - 2).bit_length()
            patches_fft = torch.fft.rfft2(
                patches,
                s=(fft_h, fft_w),
                dim=(-2, -1)
            )

            weights = self.psfs.to(device=x.device, dtype=x.dtype).unsqueeze(0)
            weights_fft = torch.fft.rfft2(
                weights,
                s=(fft_h, fft_w),
                dim=(-2, -1)
            )

            blurred_fft = patches_fft * weights_fft
            blurred = torch.fft.irfft2(
                blurred_fft,
                s=(fft_h, fft_w),
                dim=(-2, -1),
            )[
                ..., self.kernel_size - 1:self.kernel_size - 1 + height, self.kernel_size - 1:self.kernel_size - 1 + width]
        elif convolution_method == 'conv2d':
            weights = self.psfs.to(device=x.device, dtype=x.dtype).unsqueeze(1)
            blurred = F.conv2d(
                patches,
                weights.flip((-2, -1)),
                groups=channels,
                padding=0,
            )
        else:
            raise ValueError("Convolution method must be either fft or conv2d")

        if blurred.shape[-2:] != (height, width):
            raise RuntimeError("PSF convolution produced an unexpected image size.")

        # Integrate each sensor pixel over its corresponding high-resolution area.
        views = F.avg_pool2d(
            blurred,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor)
        if views.shape[-2:] != (self.lens_out_size, self.lens_out_size):
            raise RuntimeError("Sensor integration produced an unexpected size.")

        # Arrange the N lens views into a repeat_cnt x repeat_cnt mosaic.
        views = views.reshape(
            batch_size,
            self.repeat_cnt,
            self.repeat_cnt,
            channels,
            self.lens_out_size,
            self.lens_out_size,
        )
        views = views.permute(0, 3, 1, 4, 2, 5).contiguous()
        return views.reshape(
            batch_size,
            channels,
            self.output_size,
            self.output_size,
        )

    def pinv(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the regularized least-squares inverse using conjugate gradients."""
        if y.ndim != 4:
            raise ValueError(f"Expected BCHW measurement, got {tuple(y.shape)}.")
        batch_size, channels, height, width = y.shape

        if batch_size != 1:
            raise ValueError("Current CG implementation supports batch size 1.")
        if (channels, height, width) != (
                self.channels,
                self.output_size,
                self.output_size,
        ):
            raise ValueError("Measurement shape does not match the TOMBO output shape.")
        if self.forward_transpose is None:
            raise RuntimeError("This TOMBO operator was initialized with build_pinv=False.")

        with torch.no_grad():
            rhs = self.forward_transpose(y)

            def normal_operator(x: torch.Tensor) -> torch.Tensor:
                return self.forward_transpose(self.forward(x)) + self.cg_rho * x

            return cg(
                normal_operator,
                rhs,
                rtol=self.cg_rtol,
                max_iter=self.cg_max_iter,
            )


# -----------------------------------------------------------------------------
# Diffusion sampler and workflow orchestration
# -----------------------------------------------------------------------------


class Diffusion:
    """Run full simulations, forward-only measurements, or reverse reconstructions."""

    def __init__(self, args, config, device: torch.device | str | None = None):
        self.args = args
        self.config = config
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1.0 - self.betas
        alpha_bar = alphas.cumprod(dim=0)
        alpha_bar_previous = torch.cat(
            [self.betas.new_ones(1), alpha_bar[:-1]],
            dim=0,
        )
        self.alphas_cumprod_prev = alpha_bar_previous
        posterior_variance = (self.betas * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar))

        if self.model_var_type == "fixedlarge":
            self.logvar = self.betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        else:
            raise ValueError(f"Unsupported model variance type: {self.model_var_type}")

    # ------------------------------------------------------------------
    # Top-level dispatch and model loading
    # ------------------------------------------------------------------

    def sample(self) -> None:
        """Dispatch the workflow selected by ``args.mode``."""

        mode = str(getattr(self.args, "mode", "simulate")).lower()
        if mode not in VALID_MODES:
            choices = ", ".join(sorted(VALID_MODES))
            raise ValueError(f"Unknown mode {mode!r}. Choose one of: {choices}.")

        if mode == "forward":
            self.run_forward()
            return

        model, classifier_fn = self._load_diffusion_model()
        print(
            f"Run TOMBO {mode} mode. "
            f"{self.config.time_travel.T_sampling} sampling steps; "
            f"travel_length={self.config.time_travel.travel_length}; "
            f"travel_repeat={self.config.time_travel.travel_repeat}. "
        )

        if mode == "reverse":
            self.run_reverse(model, classifier_fn)
        else:
            self.run_simulation(model, classifier_fn)

    def _load_diffusion_model(self):
        """Load the configured diffusion model and optional classifier guidance."""
        classifier_fn = None

        if self.config.model.type == "simple":
            model = Model(self.config)
            dataset_name = self.config.data.dataset

            if dataset_name == "CIFAR10":
                checkpoint_name = "cifar10"
            elif dataset_name == "LSUN":
                checkpoint_name = f"lsun_{self.config.data.category}"
            elif dataset_name == "CelebA_HQ":
                checkpoint_name = "celeba_hq"
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            if checkpoint_name == "celeba_hq":
                checkpoint = os.path.join(
                    self.args.exp,
                    "logs/celeba/celeba_hq.ckpt",
                )
                if not os.path.exists(checkpoint):
                    download(
                        "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt",
                        checkpoint,
                    )
            else:
                checkpoint = get_ckpt_path(
                    f"ema_{checkpoint_name}",
                    prefix=self.args.exp,
                )

            model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            model.to(self.device).eval()
            return torch.nn.DataParallel(model), classifier_fn

        if self.config.model.type != "openai":
            raise ValueError(f"Unsupported model type: {self.config.model.type}")

        model = create_model(**vars(self.config.model))
        if self.config.model.use_fp16:
            model.convert_to_fp16()

        image_size = int(self.config.data.image_size)
        if self.config.model.class_cond:
            diffusion_checkpoint = os.path.join(
                self.args.exp,
                f"logs/imagenet/{image_size}x{image_size}_diffusion.pt",
            )
            if not os.path.exists(diffusion_checkpoint):
                download(
                    f"https://openaipublic.blob.core.windows.net/diffusion/jul-2021/{image_size}x{image_size}_diffusion_uncond.pt",
                    diffusion_checkpoint,
                )
        else:
            diffusion_checkpoint = os.path.join(
                self.args.exp,
                "logs/imagenet/256x256_diffusion_uncond.pt",
            )
            if not os.path.exists(diffusion_checkpoint):
                download(
                    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt",
                    diffusion_checkpoint,
                )

        model.load_state_dict(
            torch.load(diffusion_checkpoint, map_location=self.device)
        )
        model.to(self.device).eval()
        model = torch.nn.DataParallel(model)

        # Kept for compatibility with the original guided-diffusion setup.
        # The current DDNM reverse loop does not invoke classifier guidance.
        if self.config.model.class_cond:
            classifier_checkpoint = os.path.join(
                self.args.exp,
                f"logs/imagenet/{image_size}x{image_size}_classifier.pt",
            )
            if not os.path.exists(classifier_checkpoint):
                download(
                    f"https://openaipublic.blob.core.windows.net/diffusion/jul-2021/{image_size}x{image_size}_classifier.pt",
                    classifier_checkpoint,
                )

            classifier = create_classifier(
                **args_to_dict(
                    self.config.classifier,
                    classifier_defaults().keys(),
                )
            )
            classifier.load_state_dict(
                torch.load(classifier_checkpoint, map_location=self.device)
            )
            classifier.to(self.device)
            if self.config.classifier.classifier_use_fp16:
                classifier.convert_to_fp16()
            classifier.eval()
            classifier = torch.nn.DataParallel(classifier)

            def classifier_guidance(x, timestep, labels):
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    logits = classifier(x_in, timestep)
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected = log_probs[range(len(logits)), labels.view(-1)]
                    gradient = torch.autograd.grad(selected.sum(), x_in)[0]
                    return gradient * self.config.classifier.classifier_scale

            classifier_fn = classifier_guidance

        return model, classifier_fn

    # ------------------------------------------------------------------
    # Dataset, operator, and tensor I/O helpers
    # ------------------------------------------------------------------

    def _create_tombo_operator(self, build_pinv: bool = True):
        lens_file = getattr(self.args, "lens_file", None) or DEFAULT_LENS_FILE
        target_distance_mm = float(getattr(self.config.ddnm, "target_distance_mm", 10000.0))

        tombo = TOMBOForwardOperator(
            config=self.config,
            has_noise=self.args.add_noise,
            convolution_method=self.args.convolution_method,
            lens_file=lens_file,
            device=self.device,
            build_pinv=build_pinv,
        )
        return tombo, tombo.forward, tombo.pinv if build_pinv else None

    def _get_test_loader(self, shuffle: bool = False):
        """Create the configured test subset and deterministic data loader."""
        _, test_dataset = get_dataset(self.args, self.config)

        def seed_worker(_) -> None:
            worker_seed = self.args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(int(self.args.seed))
        loader = data.DataLoader(
            test_dataset,
            batch_size=int(self.config.sampling.batch_size),
            shuffle=shuffle,
            num_workers=int(self.config.data.num_workers),
            worker_init_fn=seed_worker,
            generator=generator,
        )
        return test_dataset, loader

    def _save_forward_pair(
            self,
            sample_dir: str | Path,
            measurement: torch.Tensor,
            ground_truth: torch.Tensor,
    ) -> None:
        """Save exact normalized tensors and display-space PNG previews."""
        sample_dir = Path(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

        measurement = measurement.detach().float().cpu()
        ground_truth = ground_truth.detach().float().cpu()
        torch.save(measurement, sample_dir / "measurement.pt")
        torch.save(ground_truth, sample_dir / "ground_truth.pt")

        tvu.save_image(
            inverse_data_transform(self.config, measurement),
            sample_dir / "measurement.png",
        )
        tvu.save_image(
            inverse_data_transform(self.config, ground_truth),
            sample_dir / "ground_truth.png",
        )

    @staticmethod
    def _load_tensor_file(path: str | Path) -> torch.Tensor:
        """Load a saved tensor on CPU and normalize it to CHW layout."""
        tensor = torch.load(path, map_location="cpu", weights_only=True)

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected a tensor in {path}, got {type(tensor).__name__}.")
        tensor = tensor.float()

        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(f"Expected batch size 1 in {path}, got {tensor.shape[0]}.")
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"Expected a CHW or 1CHW tensor in {path}, got {tuple(tensor.shape)}.")
        return tensor

    def _load_pair_tensor(
            self,
            sample_dir: str | Path,
            tensor_name: str,
            image_name: str,
    ) -> torch.Tensor:
        """Load a pair tensor, falling back to its quantized PNG preview."""
        sample_dir = Path(sample_dir)
        tensor_path = sample_dir / tensor_name
        if tensor_path.is_file():
            return self._load_tensor_file(tensor_path)

        image_path = sample_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing both {tensor_name} and {image_name} in {sample_dir}.")

        image = read_image(str(image_path)).float().div_(255.0).unsqueeze(0)
        return data_transform(self.config, image)[0].cpu()

    @staticmethod
    def _contains_saved_pair(directory: Path) -> bool:
        has_measurement = any(
            (directory / name).is_file()
            for name in ("measurement.pt", "measurement.png")
        )
        has_ground_truth = any(
            (directory / name).is_file()
            for name in ("ground_truth.pt", "ground_truth.png")
        )
        return has_measurement and has_ground_truth

    @staticmethod
    def _sample_directory_sort_key(path: Path):
        return (0, int(path.name)) if path.name.isdigit() else (1, path.name)

    # ------------------------------------------------------------------
    # Metrics and reporting
    # ------------------------------------------------------------------

    def _make_lpips_metric(self):
        return LearnedPerceptualImagePatchSimilarity(
            net_type="alex",
            normalize=True,
        ).to(self.device)

    @staticmethod
    def _compute_metrics(
            reference: torch.Tensor,
            estimate: torch.Tensor,
            lpips_metric,
            crop_border: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compare two display-space BCHW tensors in [0, 1]."""
        if reference.ndim == 3:
            reference = reference.unsqueeze(0)
        if estimate.ndim == 3:
            estimate = estimate.unsqueeze(0)
        if reference.shape != estimate.shape:
            raise ValueError(f"Metric image shapes differ: {tuple(reference.shape)} vs {tuple(estimate.shape)}.")

        if crop_border > 0 and min(reference.shape[-2:]) > 2 * crop_border:
            reference = reference[
                ..., crop_border:-crop_border, crop_border:-crop_border
            ]
            estimate = estimate[
                ..., crop_border:-crop_border, crop_border:-crop_border
            ]

        difference = estimate - reference
        mse = difference.square().mean()
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        relative_error = (
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(reference).clamp_min(1e-12)
        )

        if reference.shape[1] == 1:
            lpips_reference = reference.repeat(1, 3, 1, 1)
            lpips_estimate = estimate.repeat(1, 3, 1, 1)
        elif reference.shape[1] == 3:
            lpips_reference = reference
            lpips_estimate = estimate
        else:
            raise ValueError("LPIPS evaluation requires 1-channel or 3-channel images.")

        with torch.no_grad():
            lpips_value = lpips_metric(lpips_reference, lpips_estimate)
        return {
            "mse": mse,
            "psnr": psnr,
            "lpips": lpips_value,
            "norm_error": relative_error,
        }

    @staticmethod
    def _metric_values(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            name: float(value.detach().cpu())
            for name, value in metrics.items()
        }

    @staticmethod
    def _write_metrics(
            path: str | Path,
            sections: dict[str, dict[str, float]],
            sample_count: int | None = None,
    ) -> None:
        with open(path, "w", encoding="utf-8") as output_file:
            if sample_count is not None:
                output_file.write(f"Number of samples={sample_count}\n\n")
            for section_name, metrics in sections.items():
                output_file.write(f"{section_name}:\n")
                for metric_name, value in metrics.items():
                    output_file.write(f"  {metric_name.upper()}={value:.8g}\n")
                output_file.write("\n")

    @staticmethod
    def _accumulate_metrics(totals: dict[str, float], metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            totals[name] += value

    # ------------------------------------------------------------------
    # Forward-only and reverse-only workflows
    # ------------------------------------------------------------------

    def run_forward(self) -> None:
        """Generate and save TOMBO measurement/ground-truth pairs."""
        _, test_loader = self._get_test_loader(shuffle=False)
        _, forward_operator, _ = self._create_tombo_operator(build_pinv=False)

        output_root = Path(self.args.image_folder).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        sample_index = 0

        print(f"Forward-only mode: saving pairs to {output_root}")
        with torch.no_grad():
            for images, _ in tqdm.tqdm(test_loader, desc="TOMBO forward"):
                ground_truth_batch = data_transform(
                    self.config,
                    images.to(self.device),
                )
                measurement_batch = forward_operator(ground_truth_batch)
                if self.args.add_noise:
                    measurement_batch = get_gaussian_noisy_img(
                        measurement_batch,
                        2.0 * self.args.sigma_y,
                    )

                for batch_index in range(ground_truth_batch.shape[0]):
                    sample_dir = output_root / str(sample_index)
                    self._save_forward_pair(
                        sample_dir,
                        measurement_batch[batch_index],
                        ground_truth_batch[batch_index],
                    )
                    sample_index += 1

        print(f"Saved {sample_index} measurement/ground-truth pair(s).")

    def run_reverse(self, model, classifier_fn=None) -> None:
        """Load saved measurement/ground-truth pairs and reconstruct each one."""
        del classifier_fn  # The current DDNM reverse loop does not use it.

        input_root = Path(self.args.image_folder).resolve()
        if not input_root.is_dir():
            raise FileNotFoundError(f"Reverse input directory does not exist: {input_root}")

        if self._contains_saved_pair(input_root):
            sample_dirs = [input_root]
        else:
            sample_dirs = sorted([child
                                  for child in input_root.iterdir()
                                  if child.is_dir() and self._contains_saved_pair(child)],
                                 key=self._sample_directory_sort_key)

        if not sample_dirs:
            raise FileNotFoundError(
                "No measurement/ground-truth pairs found. Run --mode forward first, "
                "or provide measurement.pt and ground_truth.pt (PNG previews are "
                f"also accepted) under {input_root}."
            )

        _, forward_operator, inverse_operator = self._create_tombo_operator(build_pinv=True)
        lpips_metric = self._make_lpips_metric()
        metric_totals = {name: 0.0 for name in ("mse", "psnr", "lpips", "norm_error")}

        for sample_dir in tqdm.tqdm(sample_dirs, desc="TOMBO reverse"):
            measurement = self._load_pair_tensor(
                sample_dir,
                "measurement.pt",
                "measurement.png",
            ).unsqueeze(0).to(self.device)
            ground_truth = self._load_pair_tensor(
                sample_dir,
                "ground_truth.pt",
                "ground_truth.png",
            ).unsqueeze(0).to(self.device)

            self._validate_pair_shapes(sample_dir, measurement, ground_truth)
            reconstruction = self.reverse(
                forward_operator,
                inverse_operator,
                measurement,
                model,
            ).to(self.device)

            ground_truth_vis = inverse_data_transform(self.config, ground_truth[0])
            measurement_vis = inverse_data_transform(self.config, measurement[0])
            reconstruction_vis = inverse_data_transform(self.config, reconstruction[0])

            metrics = self._compute_metrics(
                ground_truth_vis.unsqueeze(0).to(self.device),
                reconstruction_vis.unsqueeze(0).to(self.device),
                lpips_metric,
            )
            metric_values = self._metric_values(metrics)
            self._accumulate_metrics(metric_totals, metric_values)

            tvu.save_image(measurement_vis, sample_dir / "measurement.png")
            tvu.save_image(ground_truth_vis, sample_dir / "ground_truth.png")
            tvu.save_image(reconstruction_vis, sample_dir / "ddnm.png")
            self._write_metrics(
                sample_dir / "errors.txt",
                {"Reconstruction compared with ground truth": metric_values},
            )

        sample_count = len(sample_dirs)
        metric_averages = {
            name: value / sample_count
            for name, value in metric_totals.items()
        }
        self._write_metrics(
            input_root / "reverse_errors.txt",
            {"Average reconstruction metrics": metric_averages},
            sample_count=sample_count,
        )

        print(f"Reconstructed {sample_count} sample(s) from {input_root}")
        print("Average metrics: " + ", ".join(f"{name}={value:.6g}" for name, value in metric_averages.items()))

    def _validate_pair_shapes(
            self,
            sample_dir: Path,
            measurement: torch.Tensor,
            ground_truth: torch.Tensor,
    ) -> None:
        expected_ground_truth = (
            1,
            int(self.config.data.channels),
            int(self.config.data.image_size),
            int(self.config.data.image_size),
        )
        expected_measurement = (
            1,
            int(self.config.data.channels),
            int(self.config.ddnm.output_size),
            int(self.config.ddnm.output_size),
        )
        if tuple(ground_truth.shape) != expected_ground_truth:
            raise ValueError(
                f"Ground truth in {sample_dir} has shape {tuple(ground_truth.shape)}; "
                f"expected {expected_ground_truth}."
            )
        if tuple(measurement.shape) != expected_measurement:
            raise ValueError(
                f"Measurement in {sample_dir} has shape {tuple(measurement.shape)}; "
                f"expected {expected_measurement}."
            )

    # ------------------------------------------------------------------
    # Full simulation workflow
    # ------------------------------------------------------------------

    def run_simulation(self, model, classifier_fn=None) -> None:
        """Run the original end-to-end forward and DDNM reconstruction test."""
        self.simplified_ddnm_plus(model, classifier_fn)

    def degradate(self, ground_truth: torch.Tensor, forward_operator, inverse_operator):
        """Generate a measurement and its regularized pseudoinverse estimate."""
        measurement = forward_operator(ground_truth)
        if self.args.add_noise:
            # sigma_y is configured in [0, 1] image units; model uses [-1, 1].
            measurement = get_gaussian_noisy_img(
                measurement,
                2.0 * self.args.sigma_y,
            )
        initial_estimate = inverse_operator(measurement)
        return measurement, initial_estimate

    def simplified_ddnm_plus(self, model, classifier_fn=None) -> None:
        """Run full simulation and report reconstruction and A^+y metrics."""
        del classifier_fn  # Kept in the interface for compatibility.

        test_dataset, test_loader = self._get_test_loader(shuffle=True)
        print(f"Dataset has size {len(test_dataset)}")

        _, forward_operator, inverse_operator = self._create_tombo_operator(
            build_pinv=True
        )
        lpips_metric = self._make_lpips_metric()
        output_root = Path(self.args.image_folder).resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        metric_names = ("mse", "psnr", "lpips", "norm_error")
        totals = {
            "Reconstructed": {name: 0.0 for name in metric_names},
            "Apy": {name: 0.0 for name in metric_names},
        }

        sample_index = 0
        initial_index = sample_index

        progress = tqdm.tqdm(test_loader, desc="TOMBO simulation")
        for images, _ in progress:
            if images.shape[0] != 1:
                raise ValueError(
                    "The current CG/pseudoinverse implementation supports "
                    "sampling.batch_size=1."
                )

            ground_truth = data_transform(self.config, images.to(self.device))
            measurement, initial_estimate = self.degradate(
                ground_truth,
                forward_operator,
                inverse_operator,
            )
            reconstruction = self.reverse(
                forward_operator,
                inverse_operator,
                measurement,
                model,
            )

            sample_dir = output_root / str(sample_index)
            self._save_forward_pair(
                sample_dir,
                measurement[0],
                ground_truth[0],
            )

            ground_truth_vis = inverse_data_transform(self.config, ground_truth[0])
            initial_estimate_vis = inverse_data_transform(
                self.config,
                initial_estimate[0],
            )
            reconstruction_vis = inverse_data_transform(
                self.config,
                reconstruction[0],
            )

            tvu.save_image(initial_estimate_vis, sample_dir / "Apy.png")
            tvu.save_image(reconstruction_vis, sample_dir / "ddnm.png")

            reconstructed_metrics = self._metric_values(
                self._compute_metrics(
                    ground_truth_vis.unsqueeze(0).to(self.device),
                    reconstruction_vis.unsqueeze(0).to(self.device),
                    lpips_metric,
                )
            )
            initial_metrics = self._metric_values(
                self._compute_metrics(
                    ground_truth_vis.unsqueeze(0).to(self.device),
                    initial_estimate_vis.unsqueeze(0).to(self.device),
                    lpips_metric,
                )
            )

            self._accumulate_metrics(totals["Reconstructed"], reconstructed_metrics)
            self._accumulate_metrics(totals["Apy"], initial_metrics)
            self._write_metrics(
                sample_dir / "errors.txt",
                {
                    "Reconstructed": reconstructed_metrics,
                    "Apy": initial_metrics,
                },
            )

            sample_index += 1
            completed = sample_index - initial_index
            progress.set_postfix(
                lpips=f"{totals['Reconstructed']['lpips'] / completed:.4f}"
            )

        sample_count = sample_index - initial_index
        if sample_count == 0:
            raise RuntimeError("The selected dataset subset contained no samples.")

        averages = {
            section: {
                name: value / sample_count
                for name, value in metrics.items()
            }
            for section, metrics in totals.items()
        }
        self._write_metrics(
            output_root / "errors.txt",
            averages,
            sample_count=sample_count,
        )

        print(f"Completed {sample_count} sample(s).")
        print(
            "Average reconstruction PSNR: "
            f"{averages['Reconstructed']['psnr']:.2f} dB"
        )
        print(
            "Average reconstruction LPIPS: "
            f"{averages['Reconstructed']['lpips']:.4f}"
        )

    # ------------------------------------------------------------------
    # DDNM reverse diffusion sampler
    # ------------------------------------------------------------------

    def reverse(self, forward_operator, inverse_operator, measurement, model):
        """Run the DDNM+ reverse-diffusion loop for one measurement."""
        sigma_y = 2.0 * float(self.args.sigma_y)
        batch_size = measurement.shape[0]

        x = torch.randn(
            batch_size,
            int(self.config.data.channels),
            int(self.config.data.image_size),
            int(self.config.data.image_size),
            device=self.device,
        )

        sampling_steps = int(self.config.time_travel.T_sampling)
        skip = self.config.diffusion.num_diffusion_timesteps // sampling_steps
        if skip <= 0:
            raise ValueError(
                "T_sampling must not exceed num_diffusion_timesteps."
            )

        times = get_schedule_jump(
            sampling_steps,
            int(self.config.time_travel.travel_length),
            int(self.config.time_travel.travel_repeat),
        )
        time_pairs = list(zip(times[:-1], times[1:]))
        x0_predictions = []
        states = [x]
        eta = float(self.args.eta)
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must be between 0 and 1.")

        with torch.no_grad():
            for current_step, next_step in time_pairs:
                current_step *= skip
                next_step *= skip
                if next_step < 0:
                    next_step = -1

                if next_step < current_step:
                    timestep = torch.full(
                        (batch_size,),
                        current_step,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    next_timestep = torch.full(
                        (batch_size,),
                        next_step,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    alpha = compute_alpha(self.betas, timestep.long())
                    alpha_next = compute_alpha(self.betas, next_timestep.long())
                    x_t = states[-1].to(self.device)

                    noise_prediction = model(x_t, timestep)
                    if noise_prediction.shape[1] == 6:
                        noise_prediction = noise_prediction[:, : self.config.data.channels]

                    x0 = (x_t - noise_prediction * (1.0 - alpha).sqrt()) / alpha.sqrt()

                    sigma_t = (1.0 - alpha_next).sqrt()
                    if sigma_t >= alpha_next * sigma_y:
                        lambda_t = 1.0
                        gamma_t = (sigma_t.square() - (alpha_next * sigma_y).square()).sqrt()
                    else:
                        lambda_t = sigma_t / (alpha_next * sigma_y)
                        gamma_t = 0.0

                    x0_projected = x0 - lambda_t * inverse_operator(forward_operator(x0) - measurement)
                    x_next = alpha_next.sqrt() * x0_projected + gamma_t * (
                            eta * torch.randn_like(x0) + math.sqrt(1.0 - eta ** 2) * noise_prediction)

                    x0_predictions.append(x0.detach().cpu())
                    states.append(x_next.detach().cpu())
                else:
                    next_timestep = torch.full(
                        (batch_size,),
                        next_step,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    alpha_next = compute_alpha(self.betas, next_timestep.long())
                    x0 = x0_predictions[-1].to(self.device)
                    x_next = (
                            alpha_next.sqrt() * x0
                            + torch.randn_like(x0) * (1.0 - alpha_next).sqrt()
                    )
                    states.append(x_next.detach().cpu())

        return states[-1]


# -----------------------------------------------------------------------------
# RePaint time-travel schedule
# -----------------------------------------------------------------------------


def get_schedule_jump(
        sampling_steps: int,
        travel_length: int,
        travel_repeat: int,
) -> list[int]:
    """Create the RePaint schedule with forward time-travel jumps."""
    if sampling_steps <= 0:
        raise ValueError("sampling_steps must be positive.")
    if travel_length <= 0:
        raise ValueError("travel_length must be positive.")
    if travel_repeat <= 0:
        raise ValueError("travel_repeat must be positive.")

    jumps = {
        step: travel_repeat - 1
        for step in range(0, sampling_steps - travel_length, travel_length)
    }

    timestep = sampling_steps
    times = []
    while timestep >= 1:
        timestep -= 1
        times.append(timestep)

        if jumps.get(timestep, 0) > 0:
            jumps[timestep] -= 1
            for _ in range(travel_length):
                timestep += 1
                times.append(timestep)

    times.append(-1)
    _check_times(times, -1, sampling_steps)
    return times


def _check_times(times: list[int], minimum: int, maximum: int) -> None:
    """Validate that a RePaint schedule has unit-sized steps and valid bounds."""
    if len(times) < 2 or times[0] <= times[1]:
        raise ValueError(f"Invalid schedule start: {times[:2]}")
    if times[-1] != -1:
        raise ValueError(f"Schedule must end at -1, got {times[-1]}.")

    for previous, current in zip(times[:-1], times[1:]):
        if abs(previous - current) != 1:
            raise ValueError(f"Schedule step must be 1: {previous} -> {current}.")
    for timestep in times:
        if not minimum <= timestep <= maximum:
            raise ValueError(f"Timestep {timestep} lies outside [{minimum}, {maximum}].")
