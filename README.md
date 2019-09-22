# creating_training_set
Fish out sequences/structures from PDB which are similar but not identical to the benchmark/test sets for training deep learning models.
I am using PSI-BLAST against all PDB protein sequences to search for pdbids <= threshold sequence identity
# test sets
Protein-protein
Protein-DNA
Protein-RNA
should have both unbound and bound conformations (to account for conformational changes)

# to add #
within identical/similar sequences different structures to create decoys

similarity measure TM-score or RMSD
