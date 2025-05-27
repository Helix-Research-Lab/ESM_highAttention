# ESM_highAttention

This repository contains code and resources for finding high attention sites from the ESM attention matrices. 

![overview](https://github.com/user-attachments/assets/e7c6bfe2-c5a7-42f5-8139-e3ac475fe1c3)

0. environment.yml file contains the package requirements
1. ESM_matrices.py
    - get the attention matrices and representation matrix from the ESM model
2. ESM_ha_loc.py 
    - Load the attention matrices (from ESM) and calculate the column sum and normalized heatmaps (provided in Zenodo)
    - Use the heatmaps to identify the convergence layer and the High Attention (HA) sites. Save the protein to convergence layer and HA site dictionaries       to data folder (provided in Github). **.pkl files should be read in as binary
3. ESM_ks.py
    - Perform the KS test per family. For each family, calculate the true family distance for each distance measure (ha_based, mean, max, cls) and 1000 
    random iterations, and calculates the KS statistics (provided in Github and Zenodo).
4. ESM_partial_dist.py
    - Calculates the pairwise distance matrices for each distance measure to perform the silhouette score analysis (results in Github).
5. ESM_spatial_dist.py
    - Takes in the pdbs files and the high attention sites and active sites to calculate the spatial distances between the HA sites and active sites 
    (results in Github).
   
