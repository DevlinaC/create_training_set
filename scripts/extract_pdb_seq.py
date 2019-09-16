"""
extracts particular sequences from the pdb_seqres.txt file
This is the test set

"""

import sys
import re
from pathlib import Path
import pathlib as pl
import itertools as itt

main_dir = pl.Path('/home/saveliy/create_training_set')
pdb_seqs_dir = main_dir / "protein_seqs_pdb"
dataset_dir = main_dir / "protein_DNA_RNA_benchmarks"
input_file = dataset_dir/ "bound_unbound_jet2dna.txt"
pdb_seqres = pdb_seqs_dir / "pdb_seqres.txt"
lst_bound = []
lst_unbound = []

with open(input_file) as f:
    for line in f:
        line=line.strip('\n')
        (bound,unbound)=line.split(',')
        #print(bound,unbound)
        lst_bound.append(bound)
        lst_unbound.append(unbound)

pattern = re.compile(r'>(?P<pdb>\w{4}[_]\w{1,4})\s.*?')
seq_dict = {}
with open(pdb_seqres) as oF:
    # only works if two number of lines is even
    for pdb_line, seq_line in itt.zip_longest(*[oF]*2):
        m = pattern.match(pdb_line)
        if m is not None:
            k = m.group('pdb')
            seq_dict[k]=seq_line.strip()
        else:
           print (f"wrong pdb line {pdb_line.strip()}")

Fileout= dataset_dir / "test_set.fasta"
FOut = open(Fileout,"w+")

for i in lst_bound:
    (pdbid,chain)=i.split('_')
    chlist = list(chain)
    for ch in chlist:
        headerline=">" + pdbid + "_" + ch + "\n"
        key = pdbid + "_" + ch
        FOut.write(headerline)
        if key in seq_dict.keys():
            FOut.write(seq_dict[key]+"\n")

for i in lst_unbound:
    (pdbid,chain)=i.split('_')
    chlist = list(chain)
    for ch in chlist:
        headerline=">" + pdbid + "_" + ch + "\n"
        key = pdbid + "_" + ch
        FOut.write(headerline)
        if key in seq_dict.keys():
            FOut.write(seq_dict[key]+"\n")

FOut.close()



