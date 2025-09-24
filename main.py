import torch 
from PIL import Image
from ldm_pipeline import LDMSuperResolutionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "/lustre/scratch/client/movian/applied/users/dungnn28/phuclh/weight_cache/models/ldm-super-resolution-4x-openimages"

# load model and scheduler
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

low_res_image = Image.open("/lustre/scratch/client/movian/applied/users/dungnn28/phuclh/minKDS/content/sr.png")
low_res_image = low_res_image.resize((128, 128))

# run pipeline in inference (sample random noise and denoise)
band_width_list = [0.1, 0.3, 0.6, 1.0]
patch_size_list = [1, 2, 4, 8]
for band_width in band_width_list:
    for patch_size in patch_size_list:
        upscaled_image = pipeline(low_res_image, num_inference_steps=50, eta=1, use_KDE=True, gamma_0=0.3, num_particles=10, band_width=band_width, patch_size=patch_size).images[0]
        upscaled_image.save(f"content/ldm_kde_band_width_{band_width}_patch_size_{patch_size}.png")


upscaled_image = pipeline(low_res_image, num_inference_steps=50, eta=1, use_KDE=False).images[0]
upscaled_image.save("content/ldm_vanilla.png")