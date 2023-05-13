from diffusers import StableDiffusionPipeline
import torch
import os
import wandb
import pickle

if __name__ == "__main__":
    config = {"batch_size": 10, "n_batches": 1}
    wandb.init(project="generate", entity="latent-dire", name="generate_fake", config=config)

    # read file LOC_synset_mapping.txt that maps ILSVRC2012_synset to WordNet synset
    mapping_caption_wordnet = {}
    with open("data/LOC_synset_mapping.txt") as f:
        for line in f:
            line = line.strip().split()
            caption = " ".join(line[1:])
            mapping_caption_wordnet[line[0]] = caption

    # get all possible prompts and save them in a list
    prompts = list(mapping_caption_wordnet.values())

    # path = f"/cluster/home/{os.environ['USER']}/Latent-DIRE/data/fake/"
    # directories = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    # prompts = [prompt for prompt in prompts if prompt not in directories]
    print("prompts loaded")

    # generate fake images from these prompts
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, torch_dtype=torch.float16)  # or other model
    pipe = pipe.to(device)
    print("pipeline created")
    for prompt in prompts:
        print(f"writing the images for the prompt : {prompt}")
        # if not os.path.exists(f"data/fake/{prompt}/"):
        #     os.makedirs(f"data/fake/{prompt}/")
        for batch in range(config["n_batches"]):
            images_generated = pipe([f'A photo of {prompt}'] * config["batch_size"]).images
            for i in range(config["batch_size"]):
                image = images_generated[i]
                image.save(f"/cluster/scratch/lcolonn/sd_generated/fake_{batch*config['batch_size'] + i}.jpg")
