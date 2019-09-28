# creating_training_set
Fish out sequences/structures from PDB which are similar but not identical to the benchmark/test sets for training deep learning models.
I am using PSI-BLAST against all PDB protein sequences to search for pdbids <= threshold sequence identity

## test sets
Protein-protein
Protein-DNA
Protein-RNA
should have both unbound and bound conformations (to account for conformational changes)

## read blast output
reads blast output 

**Options**

|  Short   | Long  |  Description        |
|:---:|:-------:|:----------------------:|
| -i  | --input_file | Path to the input file |
| -o  | --out_file | Path to output file |
| -s  | --seqid | Sequence identaty cutoff |
| -h  | --help | Prints help |

**Example**
``` bash
python3 read_blast_output.py -i ../protein_DNA_RNA_benchmarks/test_set1_results.txt -o ../protein_DNA_RNA_benchmarks/test_set.txt -s 80 
```

## get decoys
check if structures are disimular from each other

**Options**

|  Short   | Long  |  Description        |
|:---:|:-------:|:----------------------:|
| -i  | --input | Path to the input file |
| -t  | --tm_path | Path to tmalign program |
| -d  | --pdb_dir | Directory to PDB files (PDB files need to be preprocessed)  |
| -s  | --seqid | Sequence identaty cutoff |
| -r  | --rmsd | RMSD  cutoff |
| -h  | --help | Prints help |

**Example**
``` bash
python3 get_decoys.py -i ../protein_DNA_RNA_benchmarks/test_set1_results -t ~/prog/TMalign -d ~/PDB/ -s 100 -r 1.0
```

## download clean PDB
download and cleans pdbfiles

**Options**

|  Short   | Long  |  Description        |
|:---:|:-------:|:----------------------:|
| -i  | --input | Path to the Blast output file |
| -d  | --pdb_dir | Directory to PDB files (PDB files need to be preprocessed)  |
| -s  | --seqid | Sequence identaty cutoff |
| -h  | --help | Prints help |

**Example**
``` bash
python3 download_clean_pdb.py -i ../protein_DNA_RNA_benchmarks/test_set1_results -d ~\PDB -s 100
```

## to add #
within identical/similar sequences different structures to create decoys
clustering with sequence/structure similarity metrics

similarity measure TM-score or RMSD
