import torch
import torch.nn.functional as F
import einops
from jaxtyping import Float

def predicted_x0_from_noise(x_t, sigma_t, noise_pred, alpha_t):
    return (x_t - sigma_t * noise_pred) / alpha_t

def predicted_noise_from_x0(x_t, predicted_x0, alpha_t, sigma_t):
    return (x_t - alpha_t * predicted_x0) / sigma_t     

def gaussian_kernel_function(u):
    return torch.exp(-u**2 / 2) 

def mean_shift_patchwise(x: Float[torch.Tensor, "N C H W"], band_width = 1., patch_size = 4):
    N, C, H, W = x.shape
    x = einops.rearrange(x, "N C (H ps1) (W ps2) -> N (C ps1 ps2) H W", ps1=patch_size, ps2=patch_size)
    dis_mat = x.unsqueeze(1) - x.unsqueeze(0) # N, N, C*ps*ps, H_, W_
    dis_mat = dis_mat.norm(dim=2)
    dis_mat = dis_mat / band_width ** 2
    dis_mat = gaussian_kernel_function(dis_mat)
    res = torch.einsum("ijhw, jdhw -> idhw", dis_mat, x) / dis_mat.sum(dim=1, keepdim=True) - x 
    res = einops.rearrange(res, "N (C ps1 ps2) H W -> N C (H ps1) (W ps2)", ps1=patch_size, ps2=patch_size)
    return res

def choose_last_latent(x: Float[torch.Tensor, "N C H W"]):
    x_ = x.reshape(x.shape[0], -1)
    current_mean = x_.mean(dim=1, keepdim=True)
    dis = x_ - current_mean 
    dis = dis.norm(dim=-1)
    choose_ind = dis.argmin(dim=0)
    return x[choose_ind].unsqueeze(0)


    




