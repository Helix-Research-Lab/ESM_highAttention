import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy.stats import gaussian_kde
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import MuscleCommandline
from Bio.Align.Applications import ClustalOmegaCommandline
from pymsaviz import MsaViz, get_msa_testdata
import torch
import pwlf
import math
from torch import nn
from Bio import AlignIO
import sys
from scipy.spatial.distance import cdist
import random
from scipy.stats import ks_2samp
from tqdm import tqdm
import sys
from scipy.spatial.distance import cosine

def get_prot_names(fasta_file):
    seqs = SeqIO.parse(fasta_file, "fasta")
    protName = []
    for seqrecord in seqs:
        name = seqrecord.id.split('|')[1]
        protName.append(name)
    return protName

def filter_protein_list(protein_list, pid_HA):
    filtered = []
    for prot in protein_list:
        if (prot in pid_HA):
            hist_filename = '../data/heatmap/{}.npy'.format(prot)
            if (os.path.exists(hist_filename)):
                filtered.append(prot)
    return filtered

def get_HA_vectors (prot, pid_HA):
    ha_vec_set = []
    if prot in pid_HA:
        ha_list = pid_HA[prot]
        hist_filename = '../data/heatmap/{}.npy'.format(prot)
        if (os.path.exists(hist_filename)):
            try:
                heatmap = np.load(hist_filename)
            except Exception as e:
                print ('{}: {}'.format(prot, e))
                quit()
            for site in ha_list:
                ha_vec_set.append(heatmap[:, site])
        return (ha_vec_set)

def get_representation_vecs(prot):
    rep_filename = f'/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot_human/representation_matrices/{prot}.pt'
    if (os.path.exists(rep_filename)):
        try:
            data = torch.load(rep_filename)
        except Exception as e:
            print ('{}: {}'.format(prot, e))
            quit()
        rep = data[0,1:-1,:]
        cls = data[0, 1, :]
        vec_mean = torch.mean(rep, 0)
        vec_max,_ = torch.max(rep, 0)
    return vec_mean, vec_max, cls
    
def get_distances(protein_list, pid_HA):
    dist_matrix = np.zeros((len(protein_list), len(protein_list)))
    dist_matrix_mean = np.zeros((len(protein_list), len(protein_list)))
    dist_matrix_max = np.zeros((len(protein_list), len(protein_list)))
    dist_matrix_cls = np.zeros((len(protein_list), len(protein_list)))
    for i in range(0, len(protein_list)-1):
        protein_i = protein_list[i]
        vecs_i = get_HA_vectors(protein_i, pid_HA)
        vec_mean_i, vec_max_i, vec_cls_i = get_representation_vecs(protein_i)

        for j in range(i+1, len(protein_list)):
            protein_j = protein_list[j]
            vecs_j = get_HA_vectors(protein_j, pid_HA)
            vec_mean_j, vec_max_j, vec_cls_j = get_representation_vecs(protein_j)

            dist = cdist(vecs_i, vecs_j, metric='cosine')  # shape: (k, m)
            # Average of best-match distances (symmetric)
            best_i_to_j = np.min(dist, axis=1).mean()
            best_j_to_i = np.min(dist, axis=0).mean()
            final_distance = (best_i_to_j + best_j_to_i) / 2
            dist_matrix[i, j] = final_distance
            dist_matrix[j, i] = final_distance  
                               
            #mean-max-cls dist
            dist_mean = cosine(vec_mean_i, vec_mean_j)
            dist_max = cosine(vec_max_i, vec_max_j)
            dist_cls = cosine(vec_cls_i, vec_cls_j)
                               
            dist_matrix_mean[i, j] = dist_mean
            dist_matrix_mean[j, i] = dist_mean
                               
            dist_matrix_max[i, j] = dist_max
            dist_matrix_max[j, i] = dist_max

            dist_matrix_cls[i, j] = dist_cls
            dist_matrix_cls[j, i] = dist_cls
            
            
    return dist_matrix, dist_matrix_mean, dist_matrix_max, dist_matrix_cls

def lower_triangle_distances(matrix):
    """Return all unique pairwise distances (i < j) from a symmetric matrix"""
    return matrix[np.tril_indices(len(matrix), k=-1)]


if len(sys.argv) != 2: ##to be able to submit multiple job, use array_index
    print("Usage: python analyze_family.py <array_index>")
    sys.exit(1)
task_index = int(sys.argv[1])
with open ('../data/family_prot.pkl', 'rb') as file:
    fam_prot = pickle.load(file)
fam_prot_cleaned = {
    k: v for k, v in fam_prot.items()
    if k and isinstance(k, str) and k.strip() and not (isinstance(k, float) and math.isnan(k)) and not (k=='nan') and len(v) >= 2
}
family_ids = sorted(fam_prot_cleaned.keys())

## set up batches
total_batches = 30
batch_size = math.ceil(len(family_ids) / total_batches)
if task_index >= total_batches:
    print(f"Error: task_index {task_index} out of range for {total_batches} batches.")
    sys.exit(1)
# Get batch for this array index
start_idx = task_index * batch_size
end_idx = min((task_index + 1) * batch_size, len(family_ids))
batch_family_ids = family_ids[start_idx:end_idx]

with open('../data/protein_highAttnSite.pkl', 'rb') as file:
    pid_HA = pickle.load(file)
num_iter = 1000

fasta_file = '../data/uniprot_human_full.fasta'
llm_output_dir = '../outputs/llm_data'
full_prot_list = get_prot_names(fasta_file)
for file in os.listdir(os.fsencode('{}/representation_matrices'.format(llm_output_dir))):
    prot_name = (str(os.fsdecode(file)).split('.')[0])
    full_prot_list.append(prot_name)
full_prot_list_filtered = filter_protein_list(full_prot_list, pid_HA)

experiment_types = ['ha_based', 'mean', 'max', 'cls']
for family in batch_family_ids:

    protein_list = fam_prot_cleaned[family]
    protein_list_filtered = filter_protein_list(protein_list, pid_HA)
    if (len(protein_list_filtered) <= 1):
        print (f'Family {family} not big enough after filtering')
        continue
    distance_matrix, distance_matrix_mean, distance_matrix_max, distance_matrix_cls = get_distances(protein_list_filtered, pid_HA)

    exp_true = {}
    exp_iterations= {}
    for exp in experiment_types:
        output_filepath = f'../data/similarity_analysis/{exp}/{family}'
        if not os.path.exists(output_filepath):
            os.makedirs(output_filepath)
        if (exp == 'ha_based'):
            distances = lower_triangle_distances(distance_matrix)
        elif (exp == 'mean'):
            distances = lower_triangle_distances(distance_matrix_mean)
        elif (exp == 'max'):
            distances = lower_triangle_distances(distance_matrix_max)
        elif (exp == 'cls'):
            distances = lower_triangle_distances(distance_matrix_cls)
        true_mean = np.mean(distances)
        exp_true[exp] = {'distances': distances, 'true_mean': true_mean}

        with open (f'{output_filepath}/family_distances.pkl', 'wb') as file:
            pickle.dump(distances, file)

        
        exp_iterations[exp] = {'random_means':[], 'ks_stats': [], 'p_values': []}

    for i in tqdm(range(num_iter)):
        candidates = list(set(full_prot_list_filtered) - set(protein_list_filtered))
        random_sample = random.sample(candidates, len(protein_list_filtered))
        distance_matrix_random, distance_matrix_random_mean, distance_matrix_random_max, distance_matrix_random_cls = get_distances(random_sample, pid_HA)
        
        for exp in experiment_types:
            output_filepath = f'../data/similarity_analysis/{exp}/{family}'
            true_distances = exp_true[exp]['distances']
            true_mean = exp_true[exp]['true_mean']

            if (exp == 'ha_based'):
                distances_random = lower_triangle_distances(distance_matrix_random)
            elif (exp == 'mean'):
                distances_random = lower_triangle_distances(distance_matrix_random_mean)
            elif (exp == 'max'):
                distances_random = lower_triangle_distances(distance_matrix_random_max)
            elif (exp == 'cls'):
                distances_random = lower_triangle_distances(distance_matrix_random_cls)


            exp_iterations[exp]['random_means'].append(np.mean(distances_random))
            
            # Perform KS test
            if (len(true_distances) == 0 or len(distances_random) == 0):
                print (f'{family} has no true or random distances!!')
                continue
            stat, p = ks_2samp(true_distances, distances_random)
            exp_iterations[exp]['ks_stats'].append(stat)
            exp_iterations[exp]['p_values'].append(p)

    for exp in experiment_types:
        output_filepath = f'../data/similarity_analysis/{exp}/{family}'

        with open(f'{output_filepath}/random_means.pkl', 'wb') as file:
            pickle.dump(exp_iterations[exp]['random_means'], file) 
        with open(f'{output_filepath}/ks_values.pkl', 'wb') as file:
            pickle.dump(exp_iterations[exp]['ks_stats'], file)
        with open(f'{output_filepath}/ks_pvalues.pkl', 'wb') as file:
            pickle.dump(exp_iterations[exp]['p_values'], file)