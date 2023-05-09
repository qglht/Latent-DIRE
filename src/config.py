from guided_diffusion.script_util import model_and_diffusion_defaults


# args for create_model from
# https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/script_util.py#L130
UNET_ARGS = [
    "image_size",
    "num_channels",
    "num_res_blocks",
    "channel_mult",
    "learn_sigma",
    "class_cond",
    "use_checkpoint",
    "attention_resolutions",
    "num_heads",
    "num_head_channels",
    "num_heads_upsample",
    "use_scale_shift_norm",
    "dropout",
    "resblock_updown",
    "use_fp16",
    "use_new_attention_order",
]

# args for create_diffusion from
# https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/script_util.py#L386
DIFFUSION_ARGS = [
    "diffusion_steps",
    "learn_sigma",
    "noise_schedule",
    "use_kl",
    "predict_xstart",
    "rescale_timesteps",
    "rescale_learned_sigmas",
    "timestep_respacing",
]

# Settings for LSUN bedroom from the flags used at
# https://github.com/openai/guided-diffusion#upsampling
LSUN_bedroom_config = dict(
    learn_sigma=True,
    noise_schedule="linear",
    model_path="models/lsun_bedroom.pt",
    attention_resolutions="32,16,8",
    class_cond=False,
    dropout=0.1,
    image_size=256,
    num_channels=256,
    num_head_channels=64,
    num_res_blocks=2,
    resblock_updown=True,
    use_fp16=True,
    use_scale_shift_norm=True,
    diffusion_steps=1000,
    rescale_timesteps=True,  # changed from False to True
    timestep_respacing="ddim20",  # changed from "" to "ddim20"
)


def get_ADM_config():
    defaults = model_and_diffusion_defaults()
    defaults.update(LSUN_bedroom_config)
    config = {key: defaults[key] for key in UNET_ARGS + DIFFUSION_ARGS}

    return config
