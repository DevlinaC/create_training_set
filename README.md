# creating_training_set
Fish out sequences/structures from PDB which are similar but not identical to the benchmark/test sets for training deep learning models.
I am using PSI-BLAST against all PDB protein sequences to search for pdbids <= threshold sequence identity
## test sets
Protein-protein
Protein-DNA
Protein-RNA
should have both unbound and bound conformations (to account for conformational changes)

## to add #
within identical/similar sequences different structures to create decoys

similarity measure TM-score or RMSD

## get decoys
check if structures are disimular from each other

**Options**
|     |         |                        |
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