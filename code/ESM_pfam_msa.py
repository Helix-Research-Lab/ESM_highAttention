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


fasta_file = '../data/uniprot_human_full.fasta'
prot_seq_dict = parse_fasta(fasta_file)

with open('../data/family_prot.pkl', 'rb') as file:
    pfam_prot_raw = pickle.load(file)
pfam_prot = {
    k: v for k, v in pfam_prot_raw.items()
    if k and isinstance(k, str) and k.strip() and not (isinstance(k, float) and math.isnan(k)) and not (k=='nan') and len(v) >= 2
}
os.makedirs(f"../data/msa", exist_ok=True)
os.makedirs(f"../data/pfam_fasta", exist_ok=True)

for pfam in pfam_prot:
    protFam_list = pfam_prot[pfam]
    if (not os.path.exists('../data/pfam_fasta/{}.fa'.format(pfam))):
        outfile = "../data/msa/{}.txt".format(pfam)

        write_family_fasta(pfam, protFam_list)
        do_msa(pfam)
        
    i = i +1