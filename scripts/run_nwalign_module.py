from pathlib import Path
import itertools as itt
import numpy as np
import re
import nwalign3 as nw

pattern = re.compile(r'>(?P<pdb>\w{4}[_]\w{1,4})\s.*?')
seq_dict = {}

input_file = "/home/saveliy/create_training_set/protein_DNA_RNA_benchmarks/test_set1.fasta"

with open(input_file) as oF:
    # only works if two number of lines is even
    for pdb_line, seq_line in itt.zip_longest(*[oF]*2):
        m = pattern.match(pdb_line)
        if m is not None:
            k = m.group('pdb')
            seq_dict[k]=seq_line.strip()
        else:
           print (f"wrong pdb line {pdb_line.strip()}")

def run_NW_align(seq1: str, seq2: str):
    # Global alignment with a specified penalty for gap open and extend #  
    out_align = nw.global_align(seq1, seq2, gap_open=-10, gap_extend=-5, match=12, matrix='BLOSUM62')    
    return out_align

def score_nwalign(nwalign_out: list):
    align1 = nwalign_out[0]
    align2 = nwalign_out[1]
    # scoring the alignment #
    nwscore = nw.score_alignment(align1,align2, gap_open=10, gap_extend=5, matrix='/home/saveliy/nwalign3-master/nwalign3-master/nwalign3/matrices/BLOSUM62')
    return(nwscore)

for i in seq_dict:
    for j in seq_dict:
        if i != j:
            seq1 = seq_dict[i]
            seq2 = seq_dict[j]
            nwalign_out = run_NW_align(seq1,seq2)
            nwscore = score_nwalign(nwalign_out)
            print(i,j,nwscore)

