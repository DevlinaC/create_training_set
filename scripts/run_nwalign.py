from pathlib import Path
import itertools as itt
import numpy as np
from subprocess import PIPE, Popen
import re

pattern = re.compile(r'>(?P<pdb>\w{4}[_]\w{1,4})\s.*?')
seq_dict = {}

input_file = "/home/saveliy/create_training_set/protein_DNA_RNA_benchmarks/test_set.fasta"
output_file = "/home/saveliy/create_training_set/protein_DNA_RNA_benchmarks/NW_seqid.txt"
nw_path = Path('../prog/NWalign')

with open(input_file) as oF:
    # only works if two number of lines is even
    for pdb_line, seq_line in itt.zip_longest(*[oF]*2):
        m = pattern.match(pdb_line)
        if m is not None:
            k = m.group('pdb')
            seq_dict[k]=seq_line.strip()
        else:
           print (f"wrong pdb line {pdb_line.strip()}")

def run_NW_align(seq1: str, seq2: str, nw_path: Path):
    args = [str(nw_path), seq1, seq2, "3"]
    # Global alignment using NW algorithm #
    with Popen(args=args, stdout=PIPE, stderr=PIPE) as proc:
        outputlines = [l.strip('\n')
                    for l in proc.stdout.read().decode().split('\n')]
    if proc.returncode == 0:
        seqid = _score_nwalign(outputlines)
    else:
        print(f"NOTE: CANNOT align {seq1} {seq2}")
        return None
    return(seqid)

def _score_nwalign(nwalign_out: list):
    # get the seqid from the output stream #
    # Sequence identity:    0.600 (=   3/   5) #
    seqid_ptr = re.compile(
        r'Sequence identity:\s+(?P<seqid>.+?)\(\=\s+.+?\)')
    for line in nwalign_out:
        match = seqid_ptr.match(line)
        if match:
            seqid = float(match.group('seqid'))    
    return(seqid)

# go through the list of pdb seqs and find seqid #
Fout = open(output_file,"w")
for pdb1 in seq_dict:
    for pdb2 in seq_dict:
        if pdb1 != pdb2:
            seq1 = seq_dict[pdb1]
            seq2 = seq_dict[pdb2]
            nwscore = run_NW_align(seq1,seq2, nw_path)
            #print(i,j,nwscore)
            Fline = pdb1 + " " + pdb2 + " " + str(nwscore) + "\n"
            Fout.write(Fline)
Fout.close()
