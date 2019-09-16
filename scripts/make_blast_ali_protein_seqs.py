import pathlib as pl
import os
import re
from subprocess import check_output

"""
Setup start and main_directory
"""

main_dir = pl.Path('/home/saveliy/create_training_set')
in_file = "test_set1.fasta"
out_file = "test_set1_results.txt"

os.chdir(main_dir)  # move to original directory

protein_dir = main_dir / 'protein_DNA_RNA_benchmarks'

def run_blast(in_file: str, out_file: str, prot_dir: pl.Path = protein_dir) -> pl.Path:
	"""
	run blast against all sequence of PDB

	"""
	pdb_seq_dir = main_dir / "protein_seqs_pdb/pdb_seqres"
	prot_seq_file = prot_dir / in_file
	blast_output = prot_dir / out_file

	# blast -query test_set.fasta -db pdb_seqres -out test_set.txt -outfmt 6 -num_iterations 10 -comp_based_stats 1 -evalue 10 #
	cmd = "psiblast" + ' -query ' + str(prot_seq_file) + ' -db ' + str(pdb_seq_dir) + " -out " + str(blast_output) + " -outfmt 6 -num_iterations 10 -comp_based_stats 1 -evalue 10"
	#print(cmd)
	out = check_output(cmd,shell=True)
	#print(out)

run_blast(str(in_file), str(out_file))	
