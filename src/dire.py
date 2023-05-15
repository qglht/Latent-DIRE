from contextlib import nullcontext
from PIL import Image
from typing import List, Union, Tuple

from diffusers import DDIMScheduler, StableDiffusionPipeline
import torch
from torch.cuda.amp import autocast
import torch.nn as nn
from torchvision.transforms.functional import pil_to_tensor
from tqdm.auto import tqdm

# if this import doesn't work, you have not installed src, see https://www.notion.so/Docs-0dabc9ae19d54649b031e94e0cb0dff9
from src.config import get_ADM_config
from guided_diffusion.dist_util import load_state_dict
from guided_diffusion.script_util import create_model_and_diffusion


class LatentDIRE(nn.Module):
    """Based on
    https://github.com/huggingface/diffusers/blob/3045fb276352681f6b9075956e599dd8ef571872/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L56.
    https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#understanding-pipelines-models-and-schedulers.
    https://github.com/tejank10/null-text-inversion
    """

    def __init__(
        self,
        device: torch.device,
        pretrained_model_name: str = "runwayml/stable-diffusion-v1-5",
        generator: torch.Generator = torch.Generator().manual_seed(1),
        use_fp16: bool = True,
        n_steps: int = 20,  # cf. section 4.1 in the DIRE paper
    ) -> None:
        super().__init__()
        self.device = device
        self.generator = generator
        self.use_fp16 = use_fp16
        self.n_steps = n_steps

        assert pretrained_model_name in [
            "CompVis/stable-diffusion-v1-4",
            "runwayml/stable-diffusion-v1-5",
            # "stabilityai/stable-diffusion-2-1", TODO: enable prediction_type=v_predict in _ddim_inversion
        ], f"Model {pretrained_model_name} not supported. Must be one of 'CompVis/stable-diffusion-v1-4', 'runwayml/stable-diffusion-v1-5'"  # , 'stabilityai/stable-diffusion-2-1'"
        self.scheduler = DDIMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")
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
        n_steps: int = None,
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
        if n_steps is None:
            n_steps = self.n_steps
        latent = self.encode(x)
        noise = self._ddim_inversion(latent, n_steps)
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

    def _ddim_inversion(self, latent: torch.Tensor, n_steps: int = None) -> torch.Tensor:
        """
        adapted from
        https://github.com/tejank10/null-text-inversion/blob/main/notebook.ipynb
        """
        if n_steps is None:
            n_steps = self.n_steps
        batch_size = latent.shape[0]
        encoder_hidden_state = self.pipe._encode_prompt(
            [""] * batch_size,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.scheduler.set_timesteps(n_steps)
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
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        adapted from
        https://github.com/huggingface/diffusers/blob/4d35d7fea3208ddf1599e90b23ee95095b280646/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L441
        """
        latent /= self.pipe.vae.config.scaling_factor
        with autocast() if self.use_fp16 else nullcontext():
            image = self.pipe.vae.decode(latent).sample
        image = image.float()

        return image

    @staticmethod
    def img_to_tensor(image: Union[str, Image.Image], size: int, use_fp16: bool = False) -> torch.Tensor:
        if type(image) == str:
            image = Image.open(image)
        image = image.resize((size, size))
        t = pil_to_tensor(image)
        t = t.to(dtype=torch.float16 if use_fp16 else torch.float32)
        t = t / 127.5 - 1.0  # [0, 255] to [-1, 1]
        t = t.unsqueeze(0)

        return t

    @staticmethod
    def tensor_to_pil(image: torch.Tensor) -> List[Image.Image]:
        """
        adapted from
        https://github.com/huggingface/diffusers/blob/716286f19ddd9eb417113e064b538706884c8e73/src/diffusers/pipelines/pipeline_utils.py#L815
        """
        # ensure there is a batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # rescale to [0, 255]
        image = ((image + 1) * 127.5).clamp(0, 255).to(dtype=torch.uint8)  # [-1, 1] to [0, 255]
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if image.shape[-1] == 1:
            # special case for grayscale (single channel) image
            pil_image = [Image.fromarray(image.squeeze(), mode="L") for image in image]
        else:
            pil_image = [Image.fromarray(image) for image in image]

        return pil_image


class ADMDIRE(nn.Module):
    def __init__(
        self,
        device: torch.device,
        model_path: str = "models/lsun_bedroom.pt",
        generator: torch.Generator = torch.Generator().manual_seed(63057),
        use_fp16: bool = False,
        n_steps: int = 20,  # cf. section 4.1 in the DIRE paper
    ) -> None:
        super().__init__()
        self.device = device
        self.generator = generator
        self.use_fp16 = use_fp16
        self.n_steps = n_steps

        config = get_ADM_config()
        config["timestep_respacing"] = f"ddim{n_steps}"
        self.unet, self.diffusion = create_model_and_diffusion(**config)
        self.unet.load_state_dict(load_state_dict(model_path, map_location="cpu"))
        self.unet.to(device)
        self.unet.convert_to_fp16()
        self.unet.eval()

        self.scheduler = DDIMScheduler(
            beta_schedule="linear",
            # beta start and end are at https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/gaussian_diffusion.py#L31
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=True,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            set_alpha_to_one=False,
            steps_offset=1,
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        n_steps: int = None,
    ) -> torch.Tensor:
        """
        Compute DIRE(x).

        Args:
            x (torch.Tensor):
                Batch of images.
        Returns:
            Tuple[torch.Tensor, ...]: DIRE(x), DIRE(z)
        """
        if n_steps is None:
            n_steps = self.n_steps
        x.to(self.device)
        noise = self._invert(x, n_steps)
        reconstruction = self._reconstruct(noise, n_steps)
        dire = torch.abs(x - reconstruction)

        return dire, reconstruction

    @torch.no_grad()
    def _invert(self, x: torch.Tensor, n_steps: int = None, return_all: bool = False) -> torch.Tensor:
        """from https://github.com/tejank10/null-text-inversion/blob/main/notebook.ipynb"""
        if n_steps is None:
            n_steps = self.n_steps
        batch_size, channels = x.shape[:2]
        self.scheduler.set_timesteps(n_steps)
        reverse_timestep_list = reversed(self.scheduler.timesteps)
        if return_all:
            xs = [x.cpu()]

        for i in tqdm(range(len(reverse_timestep_list) - 1), desc="inversion"):
            timestep = reverse_timestep_list[i]
            next_timestep = reverse_timestep_list[i + 1]

            x = self.scheduler.scale_model_input(x, timestep)
            _t = torch.tensor([timestep] * batch_size).to(self.device)
            # predict the noise residual
            with autocast() if self.use_fp16 else nullcontext():
                pred_mean_and_variance = self.unet(x, _t)
            noise_pred, _ = pred_mean_and_variance.split(channels, dim=1)

            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_next = 1 - alpha_prod_t_next

            pred_x0 = (x - beta_prod_t**0.5 * noise_pred) / (alpha_prod_t**0.5)
            # compute the previous noisy sample x_t -> x_t+1
            x = alpha_prod_t_next**0.5 * pred_x0 + beta_prod_t_next**0.5 * noise_pred
            if return_all:
                xs.append(x.cpu())

        if return_all:
            return x, xs
        return x

    @torch.no_grad()
    def _reconstruct(self, x: torch.Tensor, n_steps: int = None, return_all: bool = False) -> torch.Tensor:
        if n_steps is None:
            n_steps = self.n_steps
        self.scheduler.set_timesteps(n_steps)
        batch_size, channels = x.shape[:2]
        if return_all:
            xs = [x.cpu()]

        for t in tqdm(self.scheduler.timesteps):
            x = self.scheduler.scale_model_input(x, timestep=t)
            _t = torch.tensor([t] * batch_size).to(self.device)
            # predict the noise residual
            with autocast() if self.use_fp16 else nullcontext():
                pred_mean_and_variance = self.unet(x, _t)
            noise_pred, _ = pred_mean_and_variance.split(channels, dim=1)

            # compute the previous noisy sample x_t -> x_t-1
            x = self.scheduler.step(noise_pred, t, x).prev_sample
            if return_all:
                xs.append(x.cpu())

        if return_all:
            return x, xs
        return x

    @staticmethod
    def img_to_tensor(image: Union[str, Image.Image], size: int, use_fp16: bool = False) -> torch.Tensor:
        if type(image) == str:
            image = Image.open(image)
        image = image.resize((size, size))
        t = pil_to_tensor(image)
        t = t.to(dtype=torch.float16 if use_fp16 else torch.float32)
        t = t / 127.5 - 1.0  # [0, 255] to [-1, 1]
        t = t.unsqueeze(0)

        return t

    @staticmethod
    def tensor_to_pil(image: torch.Tensor) -> List[Image.Image]:
        """
        adapted from
        https://github.com/huggingface/diffusers/blob/716286f19ddd9eb417113e064b538706884c8e73/src/diffusers/pipelines/pipeline_utils.py#L815
        """
        # ensure there is a batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # rescale to [0, 255]
        image = ((image + 1) * 127.5).clamp(0, 255).to(dtype=torch.uint8)  # [-1, 1] to [0, 255]
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if image.shape[-1] == 1:
            # special case for grayscale (single channel) image
            pil_image = [Image.fromarray(image.squeeze(), mode="L") for image in image]
        else:
            pil_image = [Image.fromarray(image) for image in image]

        return pil_image
