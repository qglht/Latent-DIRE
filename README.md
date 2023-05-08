# Latent DIRE
Implementing the DIRE method with an auto encoder 

# Pipeline

1) Image (either real image or generated from a Generative Model, ADM or GAN)
2) Encode Image
3) Invert Image and Reconstruct it
4) Compute DIRE with the reconstructed-latent image and then with the decoder image



--------------------------------------------------------------------------------------
Project Organization
------------

    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-xy-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- Requirements file for reproducing the environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        └── data           <- Scripts to download or generate data

Sample commit