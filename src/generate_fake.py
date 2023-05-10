from diffusers import StableDiffusionPipeline
import os
import wandb

wandb.init(project="generate", entity="latent-dire", name="generate_fake")

batch_size = 5
batches = 10

# read file src/LOC_synset_mapping.txt that maps ILSVRC2012_synset to WordNet synset

mapping_caption_wordnet = {}
with open("src/LOC_synset_mapping.txt") as f:
    for line in f:
        line = line.strip().split()
        mapping_caption_wordnet[line[0]] = line[1]

# get all possible prompts and save them in a list

prompts = list(mapping_caption_wordnet.values())
print("prompts loaded")

# generate fake images from these prompts

device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")  # or other model
pipe = pipe.to(device)
print("pipeline created")
for prompt in prompts:
    print(f"writing the images for the prompt : {prompt}")
    if not os.path.exists(f"data/train/{prompt}/"):
        os.makedirs(f"data/train/{prompt}/")
    for batch in range(batches):
        images_generated = pipe([prompt] * batch_size).images
        for i in range(batch_size):
            image = images_generated[i]
            image.save(f"data/train/{prompt}/fake_{batch*batch_size+i}.jpg")
