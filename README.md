# DIRE-GVM
Implementing the DIRE method with an auto encoder 

# Pipeline

1 : Image (either real image or generated from a Generative Model, ADM or GAN)
2 : Encode Image
3 : Invert Image and Reconstruct it
4 : Compute DIRE with the reconstructed-latent image and then with the decoder image

