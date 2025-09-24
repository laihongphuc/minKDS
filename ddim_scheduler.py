import torch 
from jaxtyping import Float
from kds import (
    predicted_x0_from_noise, 
    mean_shift_patchwise, 
    predicted_noise_from_x0,
)
from typing import Union, Optional, Callable

def ddim_step(
        x_t: Float[torch.Tensor, "N C H W"], 
        noise_pred_fn: Callable[[Float[torch.Tensor, "N C H W"], Union[int, float]], Float[torch.Tensor, "N C H W"]], 
        t: Union[int, float], 
        next_t: Union[int, float],
        alphas_cumprod: Float[torch.Tensor, "T"],
        kde_lr: Optional[float] = 0.3,
        band_width: Optional[float] = 1.0,
        patch_size: Optional[int] = 4,
    ):
    """
    x_t: current latent
    noise_pred: predicted noise
    t: current timestep
    next_t: next timestep
    alphas_cumprod: alphas_cumprod of the scheduler
    kde_lr: learning rate for KDE
    """
    alpha_t = alphas_cumprod[t].sqrt()
    sigma_t = (1 - alpha_t**2).sqrt()
    noise_pred = noise_pred_fn(x_t, t)
    x_0_predicted = predicted_x0_from_noise(x_t, sigma_t, noise_pred, alpha_t)
    m_shift = mean_shift_patchwise(x_0_predicted, band_width=band_width, patch_size=patch_size)
    x_0_steer = x_0_predicted + kde_lr * m_shift
    noise_pred_kds = predicted_noise_from_x0(x_t, x_0_steer, alpha_t, sigma_t)
    alpha_prev = alphas_cumprod[next_t].sqrt()
    sigma_prev = (1 - alpha_prev**2).sqrt()
    x_t_next = alpha_prev * x_0_steer + sigma_prev * noise_pred_kds
    return x_t_next