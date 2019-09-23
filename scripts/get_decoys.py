import re
from optparse import OptionParser, OptionValueError
from pathlib import Path
from subprocess import PIPE, Popen


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


def read_input(inFile: Path) -> list:
    data = []
    with open(inFile) as oF:
        for line in oF:
            line = line.strip()
            if len(line) < 1:  # skipping empty lines
                continue
            if line.startswith("Search "):  # skipping CONVERGED lines
                continue
            pdb1, pdb2, seqid, *rest = line.split()
            try:
                data.append(
                    {'pdb1': pdb1, 'pdb2': pdb2, 'seqid': float(seqid)})
            except ValueError:
                continue
    return(data)


def run_TM_align(pdb1: Path, pdb2: Path, tm_path: Path):
    args = [str(tm_path), str(pdb1), str(pdb2)]
#        print(args)
    with Popen(args=args, stdout=PIPE, stderr=PIPE) as proc:
        outputlines = [l.strip('\n')
                       for l in proc.stdout.read().decode().split('\n')]
    if proc.returncode == 0:
        out_dict = _parse_TMalign(outputlines)
    else:
        print(f"NOTE: CANNOT align {pdb1} {pdb2}")
        return None
    return out_dict


def _parse_TMalign(output: list):
    out_dict = {}
    TMscore1_ptr = re.compile(
        r'TM-score.+?(?P<score>\d+[.]\d+).+?(Chain_1)')
    TMscore2_ptr = re.compile(
        r'TM-score.+?(?P<score>\d+[.]\d+).+?(Chain_2)')
    RMSD_ptr = re.compile(
        r'Aligned length=.*?(\d+), RMSD=.*?(?P<RMSD>\d+[.]\d+)')
    for line in output:
        match = TMscore1_ptr.match(line)
        if match:
            out_dict['tm1'] = float(match.group('score'))
        match = TMscore2_ptr.match(line)
        if match:
            out_dict['tm2'] = float(match.group('score'))
        match = RMSD_ptr.match(line)
        if match:
            out_dict['RMSD'] = float(match.group('RMSD'))
    return out_dict


if __name__ == "__main__":
    options_parser = OptionParser()
    options_parser.add_option("-i", "--input",
                              dest="input_file", type='str',
                              help="input FILE",
                              metavar="FILE",
                              action='callback',
                              callback=_check_inputFile)
    options_parser.add_option("-t", "--tm_path",
                              dest="tm_path", type='str',
                              help="path to TM FILE",
                              metavar="FILE",
                              action='callback',
                              callback=_check_inputFile)
    options_parser.add_option("-d", "--pdb_dir",
                              dest="pdb_dir", type='str',
                              help="path to PDB DIR",
                              metavar="DIR",
                              action='callback',
                              callback=_check_inputDir)
    options_parser.add_option("-s", "--seqid",
                              dest="seqcut", type='float',
                              help="sequence id cutoff",
                              metavar="FLOAT")
    options_parser.add_option("-r", "--rmsd",
                              dest="RMSDcut", type='float',
                              help="RMSD cutoff",
                              metavar="FLOAT")
    (options, args) = options_parser.parse_args()
    # print(options.__dict__)
    inFile = Path(options.input_file)
    data = read_input(inFile)
    possible_decays = []
    for coin in filter(lambda x: x['seqid'] >= options.seqcut, data):
        pdb1 = Path(options.pdb_dir)/f'{coin["pdb1"]}.pdb'
        pdb2 = Path(options.pdb_dir)/f'{coin["pdb2"]}.pdb'
        curr_data = run_TM_align(pdb1, pdb2, options.tm_path)
        if curr_data['RMSD'] >= options.RMSDcut:
            coin.update(**curr_data)
            possible_decays.append(coin)
    for decoy in possible_decays:
        str_out = f"{decoy['pdb1']} {decoy['pdb2']}"
        str_out += f"{decoy['seqid']} {decoy['RMSD']}"
        print(str_out)
