import glob
import re
from subprocess import PIPE, Popen
from pathlib import Path,PurePosixPath

lst_files = []
lst_files = glob.glob("/home/saveliy/create_training_set/PDB/chains/*pdb")

def run_TM_align(pdb1: Path, pdb2: Path, tm_path: Path):
    args = [str(tm_path), str(pdb1), str(pdb2)]
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

tm_path = Path('../prog/TMalign')
output_file = "/home/saveliy/create_training_set/PDB/TMscores_all.txt"
Fout = open(output_file, "+w")
for pdb1 in lst_files:
    for pdb2 in lst_files:
        if pdb1 != pdb2:
            curr_data = run_TM_align(pdb1, pdb2, tm_path)
            first_pdb = PurePosixPath(pdb1).stem
            second_pdb = PurePosixPath(pdb2).stem
            Fline = first_pdb + " " + second_pdb + " " + str(curr_data['tm1']) + "\n"
            Fout.write(Fline)

Fout.close()


