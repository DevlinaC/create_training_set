import itertools as itts
from pathlib import Path
from operator import itemgetter
from optparse import OptionParser, OptionValueError


def _check_inputFile(option, opt_str, value, parser):
    f_path = Path(value)
    if not f_path.is_file():
        raise OptionValueError(f"Cannot get {str(f_path)} file")
    setattr(parser.values, option.dest, Path(f_path))
    parser.values.saved_infile = True


def _check_inputDir(option, opt_str, value, parser):
    f_path = Path(value)
    if not f_path.is_dir():
        raise OptionValueError(f"Cannot get {str(f_path)} file")
    setattr(parser.values, option.dest, Path(f_path))
    parser.values.saved_infile = True


def read_sifts_file(InFile):
    def break_line(line): return line.strip().split()
    data_dict = {}
    with open(sifts_file) as oF:
        next(oF)  # skip first line
        header_line = next(oF).strip()
        keys = header_line.split()
        for coins in map(break_line, itts.islice(oF, 0, None)):
            curr_dict = {k: v for k, v in zip(keys, coins)}
            if curr_dict['SP_PRIMARY'] not in data_dict:
                data_dict[curr_dict['SP_PRIMARY']] = {}
            if curr_dict['PDB'] not in data_dict[curr_dict['SP_PRIMARY']]:
                data_dict[curr_dict['SP_PRIMARY']][curr_dict['PDB']] = []
                data_dict[curr_dict['SP_PRIMARY']
                          ][curr_dict['PDB']].append(curr_dict['CHAIN'])
    return data_dict


if __name__ == "__main__":
    options_parser = OptionParser()
    options_parser.add_option("-i", "--interaction_file",
                              dest="interaction_file", type='str',
                              help="interaction FILE",
                              metavar="FILE",
                              action='callback',
                              callback=_check_inputFile)
    options_parser.add_option("-s", "--sifts_file",
                              dest="sifts_file", type='str',
                              help="sift FILE",
                              metavar="FILE",
                              action='callback',
                              callback=_check_inputFile)
    options_parser.add_option("-o", "--out_dir",
                              dest="out_dir", type='str',
                              help="path to where save output files",
                              metavar="DIR",
                              action='callback',
                              callback=_check_inputDir)
    # args = ['-s',
    #         'C:\\Users\\Saveliy\\Projects\\PythonCodeProjects\\work_data\\uniprot_segments_observed.tsv',
    #         '-i', 'C:\\Users\\Saveliy\\Devlina\\human_hc_uniprotids.txt',
    #         '-o', 'C:\\Users\\Saveliy\\Devlina']
    (options, args) = options_parser.parse_args()
    interaction_file = Path(options.interaction_file)
    sifts_file = Path(options.sifts_file)
    out_dir = Path(options.out_dir)
    outfile_same_chains = out_dir/'complex_same_chain.txt'
    outfile_diff_chains = out_dir/'complex_different_chain.txt'

    sifts_data = read_sifts_file(sifts_file)
    def clean_line(line): return line.strip().split('|')
    dF = open(outfile_diff_chains, 'w')
    sF = open(outfile_same_chains, 'w')
    with open(interaction_file) as oF:
        for uniprot1, uniprot2 in map(clean_line, itts.islice(oF, 0, None)):
            if uniprot1 not in sifts_data:
                continue
            if uniprot2 not in sifts_data:
                continue
            pdb1 = sifts_data[uniprot1]
            pdb2 = sifts_data[uniprot2]
            common_pdb = pdb1.keys() & pdb2.keys()
            if len(common_pdb) == 0:
                continue
            for pdb in common_pdb:
                chains1 = set(pdb1[pdb])
                chains2 = set(pdb2[pdb])
                common_chains = chains1 & chains2
                if len(common_chains) > 0:
                    str_out = f"{uniprot1}|{uniprot2}:{pdb}:"
                    for ch in common_chains:
                        str_out += f"{ch} "
                    sF.write(str_out.strip()+'\n')
                diff_chains1 = chains1 - chains2
                diff_chains2 = chains2 - chains1
                if len(diff_chains1) > 0 and len(diff_chains2) > 0:
                    str_out = f"{uniprot1}|{uniprot2}:{pdb}:"
                    for ch1 in diff_chains1:
                        str_out += f"{ch1} "
                    str_out = str_out.strip()+'|'
                    for ch2 in diff_chains2:
                        str_out += f"{ch2} "
                    dF.write(str_out.strip()+'\n')
    sF.close()
    dF.close()
