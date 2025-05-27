import pickle
import numpy as np
import os
from Bio import SeqIO
import torch
import pwlf
import math

def get_prot_names(fasta_file):
    seqs = SeqIO.parse(fasta_file, "fasta")
    protName = []
    for seqrecord in seqs:
        name = seqrecord.id.split('|')[1]
        protName.append(name)
    return protName
def make_normalized_heatmaps(protName,llm_data_filepath):
## retrieve the attention matrix and do the column sum + normalization
## to create the heatmaps
    if not os.path.exists(f'../data/heatmap'):
            os.makedirs(f'../data/heatmap')

    for prot in protName:
        heatmap = []
        filename = f'{llm_data_filepath}/attention_matrices_mean_max_perLayer/{prot}.pt'
        try:
            data = torch.load(filename)
        except Exception as e:
            print (prot)
            print (e)
            continue

        for layer in range(0,33):
            attn_matrix = data[0,0,layer,1:-1,1:-1]
            col_list = torch.sum(attn_matrix, dim=0)
            col_list_norm = col_list / max(col_list)
            heatmap.append(np.round(col_list_norm.numpy(),2))
        
        np.save(f'../data/heatmap/{prot}.npy', heatmap, allow_pickle=True)
def get_LoC_HA (protName):
    pid_impLayer = {}
    pid_highAttend = {}
    
    for prot in protName:
        heatmap_filename = '../data/heatmap/{}.npy'.format(prot)
        if (os.path.exists(heatmap_filename)):
            heatmap = np.load(heatmap_filename)
            prot_length = heatmap.shape[1]
            theta_list = []
            layer_descIndices = {}
            val_data =[] 
            pred_data = []
            indices_layer = {}

            layer_highAttend = {}
            x = [i/prot_length for i in range(0, prot_length)]
            for layer in range(0,33):
                vec = heatmap[layer, :]
                sorted_indices = np.argsort(vec)
                sorted_indices_desc = sorted_indices[::-1]
                sorted_values_desc = vec[sorted_indices_desc]
                indices_layer[layer] = sorted_indices_desc

                layer_descIndices[layer] = sorted_indices_desc

                pwlf_inst = pwlf.PiecewiseLinFit(x, sorted_values_desc)
                breaks = pwlf_inst.fit(2)

                layer_highAttend[layer] = sorted_indices_desc[:math.floor(breaks[1]*prot_length)]

                y_hat = pwlf_inst.predict(x)
                slopes = pwlf_inst.slopes
                m1, m2 = float(slopes[0]), float(slopes[1])
                print (f'Layer {layer}, {m1}, {m2}')

                theta_deg = math.degrees(math.atan((m2-m1)/(1+(m2*m1))))
                theta_list.append(theta_deg)
                val_data.append(sorted_values_desc)
                pred_data.append(y_hat)

            target = 90
            differences = [abs(value - target) for value in theta_list]
            imp_layer = differences.index(min(differences))

            pid_impLayer[prot] = imp_layer
            pid_highAttend[prot] = layer_highAttend[imp_layer]

    with open('../data/protein_highAttnSite.pkl', 'wb') as file:
        pickle.dump(pid_highAttend, file)
    with open('../data/protein_LoC.pkl', 'wb') as file:
        pickle.dump(pid_impLayer, file)

fasta_file = '../data/uniprot_human_full.fasta'
llm_data_filepath = '../outputs/llm_data'
protName = get_prot_names(fasta_file)
make_normalized_heatmaps(protName, llm_data_filepath)
get_LoC_HA(protName)