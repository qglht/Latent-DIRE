from diffusers import StableDiffusionPipeline

batch_size = 50

# read file src/LOC_synset_mapping.txt that maps ILSVRC2012_synset to WordNet synset

mapping_caption_wordnet = {}
with open('src/LOC_synset_mapping.txt') as f:
    for line in f:
        line = line.strip().split()
        mapping_caption_wordnet[line[0]] = line[1]

# get all possible prompts and save them in a list

prompts = mapping_caption_wordnet.values()

# generate fake images from these prompts

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4") # or other model
for prompt in prompts:
    pipe([prompt] * batch_size)