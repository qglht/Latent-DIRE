""" Based on https://github.com/huggingface/diffusers/blob/3045fb276352681f6b9075956e599dd8ef571872/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L56.
    See also https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#understanding-pipelines-models-and-schedulers.
"""
from typing import Tuple

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    DDIMInverseScheduler,
)
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatentDIRE(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "stabilityai/stable-diffusion-2-1-base",
        steps=20,  # cf. section 4.1 in the DIRE paper
    ) -> None:
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae").to(device)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet").to(device)

        self.reconstruction_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")
        self.reconstruction_scheduler.set_timesteps(steps)
        # fh: TODO: Discuss. This config is copied from forward scheduler
        # https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/scheduler/scheduler_config.json
        self.inversion_scheduler = DDIMInverseScheduler(
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            clip_sample=False,
            num_train_timesteps=1000,
            prediction_type="v_prediction",  # fh: TODO: Discuss/Ask TAs
            set_alpha_to_one=False,
            skip_prk_steps=True,
            steps_offset=1,
        )
        self.inversion_scheduler.set_timesteps(steps)

        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder="text_encoder").to(device)
        empty_prompt = ""
        uncond_input = tokenizer(
            empty_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        self.uncond_embedding = text_encoder(uncond_input.input_ids.to(device))[0]

    def forward(
        self,
        x: torch.Tensor,
        latent_dire: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute DIRE(x).

        Args:
            x (torch.Tensor):
                Batch of images.
            return_latent_dire (bool, optional):
                Whether to return DIRE(z), where z = VAE.encode(x) is the latent representation of x.
                Defaults to True.
        Returns:
            Tuple[torch.Tensor, ...]: DIRE(x), DIRE(z) (optional), z
        """
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.mean  # fh: TODO: Use a sample instead of mean?
            latent_reconstruction = self._invert_and_reconstruct(latent)
            reconstruction = self.vae.decode(latent_reconstruction).sample
            dire = torch.abs(x - reconstruction)
            if not latent_dire:
                return dire, latent, reconstruction
            latent_dire = torch.abs(latent - latent_reconstruction)
            return dire, latent_dire, latent, reconstruction

    def _invert_and_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        x = self._steps(x, direction="invert")
        x = self._steps(x, direction="reconstruct")

        return x

    def _steps(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        batch_size = x.shape[0]
        uncond_embedding = torch.cat([self.uncond_embedding] * batch_size)
        if direction == "invert":
            scheduler = self.inversion_scheduler
        elif direction == "reconstruct":
            scheduler = self.reconstruction_scheduler
        else:
            raise ValueError("direction must be either 'invert' or 'reconstruct'")

        for t in tqdm(scheduler.timesteps, desc=direction):
            model_input = scheduler.scale_model_input(x, t)

            # predict the noise residual
            noise_pred = self.unet(model_input, t, encoder_hidden_states=uncond_embedding).sample

            # compute the next noisy sample x_t -> x_t+1 (inversion) or x_t -> x_t-1 (reconstruction)
            x = scheduler.step(noise_pred, t, x).prev_sample

        return x
