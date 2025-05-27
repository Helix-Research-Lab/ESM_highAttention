# ESM_highAttention

This repository contains code and resources for finding high attention sites from the ESM attention matrices. 

![overview](https://github.com/user-attachments/assets/e7c6bfe2-c5a7-42f5-8139-e3ac475fe1c3)

0. environment.yml file contains the package requirements
1. ESM_matrices.py
    - get the attention matrices and representation matrix from the ESM model
2. ESM_ha_loc.py 
    - Load the attention matrices (from ESM) and calculate the column sum and normalized heatmaps (provided in Zenodo)
    - Use the heatmaps to identify the convergence layer and the High Attention (HA) sites. Save the protein to convergence layer and HA site dictionaries       to data folder (provided in Github). **.pkl files should be read in as binary
