"""
    First, extracts the chains necessary for the complex (clean pdb, only ATOM records)
    usage: python3.7 extract_chain_frm_pdb.py <textfile>
    example: python3.7 extract_chain_frm_pdb_mod.py list_pdb_chains.txt
    
    input_list looks like :
    
    pdbid,chain1,chain2

    outputfiles look like:
    pdbid_chain1.pdb and pdbid_chain2.pdb

"""

import re
import itertools as itts
import gzip
from pathlib import Path

work_dir = Path("C:\\Users\\Saveliy\\Devlina")
test_tar = Path("C:\\Users\\Saveliy\\Devlina\\1a00.pdb.gz")


def read_chain_from_gunzip_file(tarFilePath: Path, chain):
    def _line_to_skip(line):
        if not line.startswith('ATOM'):
            return True
        if line[16:20] == 'HOH':
            return True
        return False
    chain_list = []
    with gzip.open(str(tarFilePath), 'rt') as f:
        for line in itts.islice(f.readlines(), 0, 800):
            line = line.strip()
            if _line_to_skip(line):
                continue
            if line[21:22].strip() == chain:
                chain_list.append(line)
    return(chain_list)


chain_l = read_chain_from_gunzip_file(test_tar, 'A')

# get chains from the pdbfile and also process the tarfile #
# write the output file #
'''
def get_chains(pdbid, chain1,chain2,tarfilePath,outputfilePath):
    # set the PATH and file names
    pdbfile = pdbid + ".pdb1.gz"
    pdbfile1 = pdbid + "_" + chain1 + ".pdb"
    pdbfile2 = pdbid + "_" + chain2 + ".pdb"
    pdbfile1Path = outputfilePath / pdbfile1
    pdbfile2Path = outputfilePath / pdbfile2
    pdbfilePath = tarfilePath / pdbfile

    coord_re = re.compile('^ATOM')

    # creating outputfiles #
    oF1 = open(pdbfile1Path,'w+')
    oF2 = open(pdbfile2Path,'w+')

    # check if file is present, then process the file #
    if pdbfilePath.is_file():
        with tarfile.open(pdbfilePath, mode='r') as tarF:
            for line in tarF:
                line=line.strip('\n')
                if coord_re.match(line) and line[16:20] != "HOH":
                    chain = line[21].strip()
                    if chain==chain1:
                        oF1.write(line,end="\n")
                    elif chain==chain2:
                        oF2.write(line,end="\n")

    oF1.write("TER\n")
    oF1.close()
    oF2.write("TER\n")
    oF2.close()

# read the input file and call the subsequent functions to do the job #

input_list = []
# set working directories #
tarfilePath = Path('/u1/home/dc1321/data/biounit_pdb_files/')
outputfilePath = Path('/u1/home/dc1321/data/pdb_chains/')

with open("test.txt") as f:
    for line in f:
        data = line.strip()
        input_list.append(data)

for inputline in input_list:
    (pdbid, chain1,chain2) = inputline.split(',')
    get_chains(pdbid, chain1,chain2,tarfilePath,outputfilePath)
'''
