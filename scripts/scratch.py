from pathlib import Path
import sys

# setting working dir
working_dir = Path('/Users/dc1321/humanPPI')
input_file1 = sys.argv[1]
input_file2 = sys.argv[2]


"""
For calculating similarity score between a pair of complexes
AB and CD
get max (AC, AD) and (BC,BD)
get average of the two values

"""
def similarity_score(protein1,protein2,protein3,protein4):
    """
    complex1 has protein1, protein2 and complex2 has protein3, protein4
    score between complex1 and complex2, should consider the scores amongst
    all four chains

    :param protein1:
    :param protein2:
    :param protein3:
    :param protein4:
    :return seqid:

    """
    seqid = 0.00
    score1 = 0.00
    score2 = 0.00
    score3 = 0.00
    score4 = 0.00

    (prot1,prot3) = sorted([protein1,protein3])
    if prot1 == prot3:
        score1 = 1.00 # when A is compared with A
    else:
        if prot1 in uni_to_uni_compare and prot3 in uni_to_uni_compare[prot1]:
            score1 = float(uni_to_uni_compare[prot1][prot3]['seqid'])

    (prot1, prot4) = sorted([protein1, protein4])
    if prot1 == prot4:
        score2 = 1.00
    else: 
        if prot1 in uni_to_uni_compare and prot4 in uni_to_uni_compare[prot1]:
            score2 = float(uni_to_uni_compare[prot1][prot4]['seqid'])

    sum1 = score1 + score2

    (prot2, prot3) = sorted([protein2, protein3])
    if prot2 == prot3:
        score3 = 1.00
    else:
        if prot2 in uni_to_uni_compare and prot3 in uni_to_uni_compare[prot2]:
            score3 = float(uni_to_uni_compare[prot2][prot3]['seqid'])

    (prot2, prot4) = sorted([protein2, protein4])
    if prot2 == prot4:
        score4 = 1.00
    else:
        if prot2 in uni_to_uni_compare and prot4 in uni_to_uni_compare[prot2]:
            score4 = float(uni_to_uni_compare[prot2][prot4]['seqid'])

    sum2  = score3 + score4
    
    
    max_sum = max(sum1,sum2)

    seqid = round((max_sum)/2,2)

    return(seqid)

# get the list of pairs of complexes that need to be compared #
inputFile1 = working_dir / input_file1
lst_complexes = []
with open(inputFile1, 'r') as iF:
    for line in iF:
        line=line.strip()
        (uniprot1,uniprot2)=line.split('|')
        str_pairs = uniprot1 + "," + uniprot2
        lst_complexes.append(str_pairs)

lst_complexes_sorted = sorted(lst_complexes)

# A0AVT1,O15205 | A0AVT1,P0CG48 #
dict_complex_pairs = {} # list we need #
for i, protein1 in enumerate(lst_complexes_sorted):
    for j,protein2 in enumerate(lst_complexes_sorted[i:]):
        pairs = protein1 + "|" + protein2
        dict_complex_pairs[pairs]=0

# get the seqids of individual chains #
# A8MXV4 {'P07108': {'seqid': '0.36', 'align_length': '87.0'}} #
inputFile2 = working_dir / input_file2

def read_alignment_file(inputFile2) -> dict:
    out_data = {}
    with open(inputFile2) as oF:
        for line in oF:
            line = line.strip()
            (uniprot1, uniprot2, seqid, alilength) = line.split(' ')
            # sorts uniprots to make sure uni1 before uni2
            uni1, uni2 = sorted([uniprot1,uniprot2])
            if uni1 not in out_data:
                out_data[uni1] = {}  # initialize inside the dictionary
            out_data[uni1][uni2] = {'seqid': seqid, 'align_length': alilength}
    return out_data

uni_to_uni_compare = read_alignment_file(inputFile2)


 # do the comparison #
for key in dict_complex_pairs.keys():
    (complex1, complex2) = key.split('|')
    if complex1 != complex2:
        (protein1,protein2) = complex1.split(',')
        (protein3,protein4) = complex2.split(',')
        seqid = similarity_score(protein1,protein2,protein3,protein4)
        out_str = protein1 + "_" + protein2 + " " + protein3 + "_" + protein4 + " " + str(seqid)
        print(out_str)
