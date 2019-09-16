import pathlib as pl
import os
import re
from subprocess import check_output

"""
Read blast output to find sequences at a threshold

"""

main_dir = pl.Path('/home/saveliy/create_training_set')
in_file = "./test_set1_results.txt"
out_file = "./training_set.txt"

protein_dir = main_dir / 'protein_DNA_RNA_benchmarks'

def read_blast(in_file: str, out_file: str, threshold: float ):
    """
	read the blast output and create training set based on threshold

	"""
    blast_output = protein_dir / in_file
    training_set = protein_dir / out_file

    FOut = open(training_set,"w")

    with open(blast_output) as infile:
        for line in infile:
            # to skip blank lines and the line stating search has CONVERGED #
            if not line.startswith('Search has CONVERGED') and line.strip():
                line = line.strip('\n')
                line_str = line.split()
                seqid = float(line_str[2])
                if seqid < threshold:
                    FOut.write(line_str[0] + "," + line_str[1] + "," + line_str[2] +"\n")
    
    FOut.close()

# set threshold for training set #
read_blast(str(in_file), str(out_file), 90 )	
