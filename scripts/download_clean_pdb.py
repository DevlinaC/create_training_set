import pathlib as pl
import os
import re
import urllib.request
import gzip

"""
Read blast output to find sequences at a threshold

"""

main_dir = pl.Path('/home/saveliy/create_training_set')
in_file = "./test_set1_results.txt"
protein_dir = main_dir  / 'protein_DNA_RNA_benchmarks'
pdb_dir = main_dir / 'PDB'
pdb_lst = []

def download_pdb(in_file: str, threshold: float ):
    """
	read the blast output and download pdbid

    """
    blast_output = protein_dir / in_file


    with open(blast_output) as infile:
        for line in infile:
            # to skip blank lines and the line stating search has CONVERGED #
            if not line.startswith('Search has CONVERGED') and line.strip():
                line = line.strip('\n')
                line_str = line.split()
                seqid = float(line_str[2])
                if seqid == threshold:
                    if line_str[0] not in pdb_lst:
                        pdb_lst.append(line_str[0])
                    if line_str[1] not in pdb_lst:
                        pdb_lst.append(line_str[1])
    
    print(pdb_lst)
    for i in pdb_lst:
        pdbid,chainid = i.split("_")
        file_name = pdbid + ".pdb"
        file_path = pdb_dir / file_name
        url = "http://www.rcsb.org/pdb/files/" + pdbid + ".pdb.gz"
        with urllib.request.urlopen(url) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                with open(file_path, 'wb') as out_file:
                    data = uncompressed.read() 
                    out_file.write(data)
        
        clean_pdb(chainid,pdbid)


def clean_pdb(chainid: str, pdbid: str):
    
    """
    
    clean the pdb file to have only ATOM records and specific chain, 
    with TER at the end
    
    """
    file_out = pdbid + "_" + chainid + ".pdb"
    file_in = pdbid + ".pdb"
    fileout_path =  pdb_dir /file_out
    filein_path = pdb_dir/ file_in
   
    coord_re = re.compile("^ATOM")
    pdbfhin = open(filein_path,'r')

    with open(fileout_path,'w+') as pdbfhout:
        for line in pdbfhin:
            if coord_re.match(line) and line[21] in chainid:
                pdbfhout.write(line + "\n")
        pdbfhout.write("TER\n")

    pdbfhin.close()


# set threshold for decoys #
download_pdb(str(in_file), 100)	
