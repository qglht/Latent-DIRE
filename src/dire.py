""" Based on 
https://github.com/huggingface/diffusers/blob/3045fb276352681f6b9075956e599dd8ef571872/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L56.
https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#understanding-pipelines-models-and-schedulers.
https://github.com/tejank10/null-text-inversion
"""

from contextlib import nullcontext
from PIL import Image
from typing import List, Union, Tuple


from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torchvision.transforms import PILToTensor
from tqdm.auto import tqdm


class LatentDIRE(nn.Module):
    def __init__(
        self,
        device: torch.device,
        pretrained_model_name: str = "CompVis/stable-diffusion-v1-4",
        generator: torch.Generator = torch.Generator().manual_seed(1),
        use_fp16: bool = False,
    ) -> None:
        super().__init__()
        self.device = device
        self.generator = generator
        self.use_fp16 = use_fp16

        assert pretrained_model_name in [
            "CompVis/stable-diffusion-v1-4",
            "runwayml/stable-diffusion-v1-5",
            # "stabilityai/stable-diffusion-2-1", TODO: enable prediction_type=v_predict in _ddim_inversion
        ], f"Model {pretrained_model_name} not supported. Must be one of 'CompVis/stable-diffusion-v1-4', 'runwayml/stable-diffusion-v1-5'"  # , 'stabilityai/stable-diffusion-2-1'"
        self.scheduler = DDIMScheduler.from_config(pretrained_model_name, subfolder="scheduler")
        self.inversion_scheduler = DDIMInverseScheduler.from_config(pretrained_model_name, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name,
            safety_checker=None,
            scheduler=self.scheduler,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
        ).to(device)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        n_steps: int = 20,  # cf. section 4.1 in the DIRE paper
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute DIRE(x).

        Args:
            x (torch.Tensor):
                Batch of images.
            n_steps (int, optional):
                Number of steps to take. Defaults to 20.
        Returns:
            Tuple[torch.Tensor, ...]: DIRE(x), DIRE(z), reconstruction, latent_reconstruction, latent
        """
        latent = self.encode(x)
        self.scheduler.set_timesteps(n_steps)
        noise = self._ddim_inversion(latent)
        batch_size = noise.shape[0]
        noise = noise.to(dtype=torch.float16 if self.use_fp16 else torch.float32)
        latent_reconstruction = self.pipe(
            prompt=[""] * batch_size,
            latents=noise,
            num_inference_steps=n_steps,
            output_type="latent",
            generator=self.generator,
        ).images
        latent_dire = torch.abs(latent - latent_reconstruction)
        reconstruction = self.decode(latent_reconstruction)
        dire = torch.abs(x - reconstruction)

        return dire, latent_dire, reconstruction, latent_reconstruction, latent

    def _ddim_inversion(self, latent: torch.Tensor) -> torch.Tensor:
        """from https://github.com/tejank10/null-text-inversion/blob/main/notebook.ipynb"""
        batch_size = latent.shape[0]
        encoder_hidden_state = self.pipe._encode_prompt(
            [""] * batch_size, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
        )
        reverse_timestep_list = reversed(self.scheduler.timesteps)

        for i in tqdm(range(len(reverse_timestep_list) - 1), desc="inversion"):
            timestep = reverse_timestep_list[i]
            next_timestep = reverse_timestep_list[i + 1]
            latent_model_input = self.scheduler.scale_model_input(latent, timestep)
            with autocast() if self.use_fp16 else nullcontext():
                noise_pred = self.pipe.unet(latent_model_input, timestep, encoder_hidden_state).sample

            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_next = 1 - alpha_prod_t_next

            pred_x0 = (latent - beta_prod_t**0.5 * noise_pred) / (alpha_prod_t**0.5)
            latent = alpha_prod_t_next**0.5 * pred_x0 + beta_prod_t_next**0.5 * noise_pred

        return latent

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with autocast() if self.use_fp16 else nullcontext():
            # fh: TODO: Use mean instead of sample?
            latent = self.pipe.vae.encode(x).latent_dist.sample(generator=self.generator)
        latent *= self.pipe.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, output_pil: bool = True) -> Union[torch.Tensor, List[Image.Image]]:
        """
        adapted from
        https://github.com/huggingface/diffusers/blob/4d35d7fea3208ddf1599e90b23ee95095b280646/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L441
        """
        latent /= self.pipe.vae.config.scaling_factor
        with autocast() if self.use_fp16 else nullcontext():
            image = self.pipe.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.float()

        return image

    @staticmethod
    def tensor_to_pil(image: torch.Tensor) -> List[Image.Image]:
        """
        adapted from
        https://github.com/huggingface/diffusers/blob/716286f19ddd9eb417113e064b538706884c8e73/src/diffusers/pipelines/pipeline_utils.py#L815
        """
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        if image.ndim == 3:
            image = image[None, ...]
        image = (image * 255).round().astype("uint8")
        if image.shape[-1] == 1:
            # special case for grayscale (single channel) image
            pil_image = [Image.fromarray(image.squeeze(), mode="L") for image in image]
        else:
            pil_image = [Image.fromarray(image) for image in image]

        return pil_image

    @staticmethod
    def img_to_tensor(image: Union[str, Image.Image], use_fp16: bool = False, size: int = 512) -> torch.Tensor:
        if type(image) == str:
            image = Image.open(image)
        image = image.resize((size, size))
        image = image.convert("RGB")
        t = PILToTensor()(image)
        t = t.to(dtype=torch.float16 if use_fp16 else torch.float32)
        t = t / 127.5 - 1.0
        t = t.unsqueeze(0)
        return t
