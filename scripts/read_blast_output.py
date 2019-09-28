import re
from optparse import OptionParser, OptionValueError
from pathlib import Path


"""
Read blast output to find sequences at a threshold
"""


def _check_inputFile(option, opt_str, value, parser):
    f_path = Path(value)
    if not f_path.is_file():
        raise OptionValueError(f"Cannot get {str(f_path)} file")
    setattr(parser.values, option.dest, Path(f_path))
    parser.values.saved_infile = True


def read_blast(in_file: Path, out_file: Path, threshold: float):
    """
    read the blast output and create training set based on threshold
    """
    with open(in_file) as iF, open(out_file, 'w') as oF:
        for line in iF:
            # to skip blank lines and the line stating search has CONVERGED #
            if not line.startswith('Search has CONVERGED') and line.strip():
                line = line.strip()
                line_str = line.split()
                seqid = float(line_str[2])
                if seqid < threshold:
                    oF.write(line_str[0] + "," +
                             line_str[1] + "," + line_str[2] + "\n")


if __name__ == "__main__":
    options_parser = OptionParser()
    options_parser.add_option("-i", "--input_file",
                              dest="input_file", type='str',
                              help="input FILE",
                              metavar="FILE",
                              action='callback',
                              callback=_check_inputFile)
    options_parser.add_option("-o", "--out_file",
                              dest="out_file", type='str',
                              help="output FILE",
                              metavar="FILE")
    options_parser.add_option("-s", "--seqid",
                              dest="seqcut", type='float',
                              help="sequence id cutoff",
                              metavar="FLOAT")
    (options, args) = options_parser.parse_args()
    in_file = Path(options.input_file)
    out_file = Path(options.out_file)
    cutoff = float(options.seqcut)
    read_blast(in_file, out_file, cutoff)
