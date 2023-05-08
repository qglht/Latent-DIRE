import scipy.io as sio
import os
import pdb

## create mapping from ILSVRC2012_val ILSVRC2012_id 


filenames = os.listdir('data/images/ILSVRC2012_img_val/')
mapping_val_id = {filename:None for filename in filenames}
pdb.set_trace()
with open('src/ILSVRC2012_validation_ground_truth.txt') as f:
    for i, line in enumerate(f):
        mapping_val_id[filenames[i]] = int(line.strip())

## create mapping from ILSVRC2012_id to ILSVRC2012_synset

# open the file src/meta.mat thanks to scipy.io 

mapping_id_caption = {}
mat_contents = sio.loadmat('src/meta.mat')
for i in range(mat_contents['synsets'].shape[0]):
    pdb.set_trace()
    mapping_id_caption[mat_contents['synsets'][i][0][0][0][0]] = mat_contents['synsets'][i][0][2][0]

## create mapping from ILSVRC2012_val to ILSVRC2012_synset

mapping_val_caption = {filename:mapping_id_caption[mapping_val_id[filename]] for filename in filenames}

print(mapping_val_caption)

# from diffusers import StableDiffusionPipeline

# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5") # or other model
# pipe([prompt] * batch_size)