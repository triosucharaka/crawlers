
from diffusers import AutoencoderKL
import imageio
import numpy as np
from PIL import Image
import torch
import einops

MODEL_NAME = "runwayml/stable-diffusion-v1-5"
PATH_TO_TENSOR = "video.pt"
FRAME_COUNT = 55
FPS_COUNT = 24

vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder='vae').to("cuda")

tmp_tensor = torch.load(PATH_TO_TENSOR).to("cuda")
print(tmp_tensor.shape)

tmp_tensor = tmp_tensor.unsqueeze(0)
tmp_tensor = einops.rearrange(tmp_tensor, 'b c f h w -> b f c h w')

frame_list = []

for frame_index in range(FRAME_COUNT):
    a_frame = tmp_tensor[:, frame_index, :, :, :]
    latents = a_frame
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    frame_list.append(pil_images[0])
    print(frame_index, "/", FRAME_COUNT)

index = 0
# for img in frame_list:
#     index += 1
#     img.save(f'/workspace/diffused-video-trainer/outimg/{index}_out.png')

imageio.mimsave('animation.gif', [np.array(img) for img in frame_list], 'GIF', fps=FPS_COUNT)
