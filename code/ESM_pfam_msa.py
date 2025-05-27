import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import operator
import collections
import torch
import seaborn as sns
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Align.Applications import MuscleCommandline
from Bio.Align.Applications import ClustalOmegaCommandline
from pymsaviz import MsaViz, get_msa_testdata
import os
import math
import pickle
from Bio import AlignIO
from collections import Counter
from pymsaviz import MsaViz, get_msa_testdata
import os

def parse_fasta(fasta_file):
    seqs = SeqIO.parse(fasta_file, "fasta")
    protName_full = []
    prot_seq_dict = {}
    for seqrecord in seqs:
        name = seqrecord.id.split('|')[1]
        protName_full.append(name)
        seq = seqrecord.seq
        prot_seq_dict[name] = seq
    return prot_seq_dict

def write_family_fasta(pfam, protFam_list):

    protFam_fasta_file = '../data/pfam_fasta/{}.fa'.format(pfam)
    protFam_fasta = open(protFam_fasta_file, "w")
    #size_list = []
    count = 0
    for protId in protFam_list:
        if (protId in prot_seq_dict):
            seq = prot_seq_dict[protId]
            #size_list.append(len(protId_seq[protId]))
            count += 1
            protFam_fasta.write(">"+str(protId)+'\n'+str(seq)+'\n')
    protFam_fasta.close()

def do_msa(pfam):
    protFam_fasta_file = '../data/pfam_fasta/{}.fa'.format(pfam)
    clustalomega_cline = ClustalOmegaCommandline(infile=protFam_fasta_file, outfile=outfile, verbose=True, auto=True, force=True)
    clustalomega_cline()  

def convert_index(seq_ind, prot, seq):
    counter = 0
    alignSeq = seq
    for align_ind in range(0, len(alignSeq)):
        c = alignSeq[align_ind]
        if (c != '-'):
            counter += 1
            if (counter == seq_ind):
                return (align_ind+1)
def get_consensus_with_percentages(alignment):
    
    alignment_length = alignment.get_alignment_length()
    num_sequences = len(alignment)
    
    consensus_seq = []
    consensus_percentages = []
    
    # Iterate through each column (position) in the alignment
    for i in range(alignment_length):
        column = alignment[:, i]
        
        # Count the occurrence of each residue in the column
        column_counter = Counter(column)
        
       # Sort residues by count in descending order
        sorted_residues = column_counter.most_common()

        # Find the first non-gap residue
        for residue, count in sorted_residues:
            if residue != '-':
                consensus_seq.append(residue)
                percentage = (count / num_sequences) * 100
                consensus_percentages.append(percentage)
                break
        else:
            # If all residues are gaps
            consensus_seq.append('-')
            consensus_percentages.append(0.0)
 
    return (consensus_seq, consensus_percentages)


fasta_file = '../data/uniprot_human_full.fasta'
prot_seq_dict = parse_fasta(fasta_file)

with open('../data/family_prot.pkl', 'rb') as file:
    pfam_prot_raw = pickle.load(file)
pfam_prot = {
    k: v for k, v in pfam_prot_raw.items()
    if k and isinstance(k, str) and k.strip() and not (isinstance(k, float) and math.isnan(k)) and not (k=='nan') and len(v) >= 2
}
with open('../data/prot_highAttnSite.pkl', 'rb') as file:
    prot_highAttend = pickle.load(file)

os.makedirs(f"../data/msa", exist_ok=True)
os.makedirs(f"../data/pfam_fasta", exist_ok=True)

consensus_HA = []
for pfam in pfam_prot:
    protFam_list = pfam_prot[pfam]
    if (not os.path.exists('../data/pfam_fasta/{}.fa'.format(pfam))):
        outfile = "../data/msa/{}.txt".format(pfam)

        write_family_fasta(pfam, protFam_list)
        do_msa(pfam)
        
        alignment = AlignIO.read(outfile, "fasta")
        consensus_sequence, consensus_percentages = get_consensus_with_percentages(alignment)
        
        for record in alignment:
            prot =record.id
            seq = str(record.seq)
            if (prot in prot_highAttend):
                for site in prot_highAttend[prot]:
                    converted_site = convert_index(site, prot, seq)
                    if (converted_site is not None):
                        consensus_HA.append(consensus_percentages[converted_site])

with open('../data/alignment_consensus_HA.pkl', 'wb') as file:
    pickle.dump(consensus_HA, file)