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


def similarity_score(complex1, complex2, score_dict):
    """
    complex1 has protein1, protein2 and complex2 has protein3, protein4
    score between complex1 and complex2, should consider the scores amongst
    all four chains

    :param complex1:
    :param complex2:
    :return seqid:

    """
    def get_score(p1, p2, m=score_dict):
        if p1 == p2:
            return 1.0
        p1, p2 = sorted([p1, p2])
        if p1 not in m:
            print(f"Cannot find pair {p1} {p2}")
            raise KeyError
        if p2 not in m[p1]:
            print(f"Cannot find pair {p1} {p2}")
            raise KeyError
        return m[p1][p2]['seqid']

    p1, p2 = complex1
    p3, p4 = complex2
    pair1 = [get_score(p1, p3), get_score(p2, p4)]
    pair2 = [get_score(p2, p3), get_score(p1, p4)]

    max_sum = max(sum(pair1), sum(pair2))
    seqid = round((max_sum)/2.0, 2)  # no need to round better to format output

    return max(sum(pair1), sum(pair2))/2.0


def read_complex_file(inFile):
    # help function to sort complexes
    def complex_key(comp): return f"{comp[0]}_{comp[1]}"
    lst_complexes = []
    with open(inFile, 'r') as iF:
        for line in iF:
            line = line.strip()
            (uniprot1, uniprot2) = line.split('|')
            str_pairs = (uniprot1,  uniprot2)
            lst_complexes.append(str_pairs)
    return sorted(lst_complexes, key=complex_key)


def read_alignment_file(inputFile2) -> dict:
    out_data = {}
    with open(inputFile2) as oF:
        for line in oF:
            line = line.strip()
            (uniprot1, uniprot2, seqid, alilength) = line.split(' ')
            # sorts uniprots to make sure uni1 before uni2
            uni1, uni2 = sorted([uniprot1, uniprot2])
            if uni1 not in out_data:
                out_data[uni1] = {}  # initialize inside the dictionary
            out_data[uni1][uni2] = {'seqid': seqid, 'align_length': alilength}
    return out_data


# get the list of pairs of complexes that need to be compared #
inputFile1 = working_dir / input_file1
complex_list = read_complex_file(inputFile1)
"""
U dont need to do this part 
# A0AVT1,O15205 | A0AVT1,P0CG48 #
dict_complex_pairs = {}  # list we need #
for i, protein1 in enumerate(lst_complexes_sorted):
    for j, protein2 in enumerate(lst_complexes_sorted[i:]):
        pairs = protein1 + "|" + protein2
        dict_complex_pairs[pairs] = 0
"""
# get the seqids of individual chains #
# A8MXV4 {'P07108': {'seqid': '0.36', 'align_length': '87.0'}} #
inputFile2 = working_dir / input_file2
uni_to_uni_compare = read_alignment_file(inputFile2)

for i, comp1 in enumerate(complex_list):
    for comp2 in complex_list[i]:
        score = similarity_score(comp1, comp2, uni_to_uni_compare)
        str_out = f"{comp1[0]}_{comp1[1]} {comp2[0]}_{comp2[1]} {score:0.2f}"
        print(str_out)
