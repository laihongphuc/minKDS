import inspect
from typing import List, Optional, Tuple, Union
from torch.nn.functional import unfold, fold

import numpy as np
import PIL.Image
import torch
import torch.utils.checkpoint
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.models import UNet2DModel, VQModel
from diffusers.utils import PIL_INTERPOLATION, is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from kds import choose_last_latent
from ddim_scheduler import ddim_step

def preprocess(image):
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    # Convert to RGB if the image has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

class LDMSuperResolutionPipeline(DiffusionPipeline):
    
    def __init__(
        self,
        vqvae: VQModel,
        unet: UNet2DModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()
        self.register_modules(vqvae=vqvae, unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        batch_size: Optional[int] = 1,
        num_inference_steps: Optional[int] = 100,
        num_particles: Optional[int] = 4,
        gamma_0: Optional[float] = 0.1,  # base steering strength
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        use_KDE: bool = True,
        band_width: Optional[float] = 1.0,
        patch_size: Optional[int] = 4,
    ) -> Union[Tuple, ImagePipelineOutput]:

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError(f"`image` has to be of type `PIL.Image.Image` or `torch.Tensor` but is {type(image)}")

        if isinstance(image, PIL.Image.Image):
            image = preprocess(image)

        height, width = image.shape[-2:]

        # in_channels should be 6: 3 for latents, 3 for low resolution image
        latents_shape = (batch_size * num_particles, self.unet.config.in_channels // 2, height, width)
        latents_dtype = next(self.unet.parameters()).dtype

        latents = randn_tensor(latents_shape, generator=generator, device=self.device, dtype=latents_dtype).requires_grad_(True)

        image = image.to(device=self.device, dtype=latents_dtype)
        image = image.repeat_interleave(num_particles, dim=0)

        # set timesteps and move to the correct device
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps_tensor = self.scheduler.timesteps

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature.
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta
        def noise_pred_fn(latents, t):
            latents_input = torch.cat([latents, image], dim=1)
            latents_input = self.scheduler.scale_model_input(latents_input, t)
            noise_pred    = self.unet(latents_input, t).sample
            return noise_pred
        
        for i, t in enumerate(self.progress_bar(timesteps_tensor)):

            if use_KDE:
              if i < len(timesteps_tensor) - 1:
                  next_t = timesteps_tensor[i + 1]
              else:
                  next_t = 0
              if t / self.scheduler.num_train_timesteps > 0.3:
                kde_lr = gamma_0 
              else:
                kde_lr = 0.0
              latents = ddim_step(latents, noise_pred_fn, t, next_t, self.scheduler.alphas_cumprod, kde_lr=kde_lr, band_width=band_width, patch_size=patch_size)

            else:
              latents_input = torch.cat([latents, image], dim=1)
              latents_input = self.scheduler.scale_model_input(latents_input, t)
              noise_pred    = self.unet(latents_input, t).sample
              out = self.scheduler.step(noise_pred, t, latents, **extra_kwargs)
              latents = out.prev_sample.detach().requires_grad_(False)
            
            if XLA_AVAILABLE:
                xm.mark_step()
        # Apply to choose the latent nearest to the mean 
        if use_KDE:
          latents = choose_last_latent(latents)
        # decode the image latents with the VQVAE
        image = self.vqvae.decode(latents).sample
        image = torch.clamp(image, -1.0, 1.0)
        image = image / 2 + 0.5
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
