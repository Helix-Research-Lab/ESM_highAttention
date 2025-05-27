import os
from Bio import SeqIO
import torch
import numpy as np
import esm
import torch
from Bio import SeqIO
import os

def parse_fasta_biopython(filepath):
    fasta_sequences = SeqIO.parse(open(filepath),'fasta')
    fasta_list = [(record.id.split('|')[1], str(record.seq)) for record in fasta_sequences]
    return fasta_list

def get_llm_data(seq,output_fp):
    if not os.path.exists(f'{output_fp}/attention_matrices_mean_max_perLayer'):
        os.makedirs(f'{output_fp}/attention_matrices_mean_max_perLayer')
    if not os.path.exists(f'{output_fp}/representation_matrices'):
        os.makedirs(f'{output_fp}/representation_matrices')
        
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    for i in range (0,len(seq)):
        name = seq[i][0]
        if (i % 10 ==0):
            print (i)
        attn_exists = os.path.exists('{}/attention_matrices_mean_max_perLayer/{}.pt'.format(output_fp, name))
        rep_exists = os.path.exists('{}/representation_matrices/{}.pt'.format(output_fp, name))
        if (not attn_exists):
            batch_labels, batch_strs, batch_tokens = batch_converter([(seq[i])])
            try:
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            except Exception as e:
                print (name)
                print (e)
                continue
            attention_matrices = results['attentions']
            attn_mean_pooled = attention_matrices.mean(dim=2, keepdim=True)
            attn_max_pooled, _ = attention_matrices.max(dim=2, keepdim=True)
            combined_attn_tensor = torch.stack((attn_mean_pooled.squeeze(2), attn_max_pooled.squeeze(2)), dim=0)

            token_representations = results["representations"][33]
    
            torch.save(combined_attn_tensor, '{}/attention_matrices_mean_max_perLayer/{}.pt'.format(output_fp, name))
            torch.save(token_representations, '{}/representation_matrices/{}.pt'.format(output_fp,name))
        elif (not rep_exists and attn_exists):
            batch_labels, batch_strs, batch_tokens = batch_converter([(seq[i])])
            try:
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33])#, return_contacts=True)
            except Exception as e:
                print (name)
                print (e)
                continue
            token_representations = results["representations"][33]
            torch.save(token_representations, '{}/representation_matrices/{}.pt'.format(output_fp,name))

fasta_file = '../data/uniprot_human_full.fasta'
seq = parse_fasta_biopython(fasta_file)
output_fp = '../outputs/llm_data'
get_llm_data(seq, output_fp)

