# TCRTransport
This is a refactoring and updating of [transport](https://github.com/matsengrp/transport/tree/main) from the Matsen lab.

## Usage
Consider two TCR repertoires stored at `FILE1` and `FILE2`, where the first repertoire is to be used as the reference repertoire.
I.e., we want to see which sequences from the second repertoire are distinct from the first.
Assume that there are the columns `'cdr3b', 'vb'` which point to the BCDR3 and TRBV gene for the TRB receptor sequence.

```python
from tcr_transport.tcr_transport import TCRTransport

# Create the TCRTransport class object.
tt = TCRTransport()

# Add the reference repertoire. The reference bool must be set to True.
tt.add_repertoire(FILE1, beta_cols=['cdr3b', 'vb'], reference=True)

# Add the repertoire which is to be compared to the reference repertoire and in which we will look for outlier sequences and clusters.
tt.add_repertoire(FILE2, beta_cols=['cdr3b', 'vb'])

# Compute the sample enrichment on the sample sequences.
tt.compute_sample_enrichment()

# Compute the p values of the observed enrichment scores using a randomization test.
tt.compute_significance()

# Create clusters of TCRs using the enrichment score to identify the focal sequence and TCRdist to obtain neighbors
tt.create_clusters()
```

If paired-chain data is available, you can point to the columns containg the TRAV gene and ACDR3 sequence. Here we assume, they are pointed to by `'cdr3a', 'va'` (of course, this will depend on the format of your data).

```python
from tcr_transport.tcr_transport import TCRTransport

# Create the TCRTransport class object. We increase the neighbor_radius from the default of 48 since TCRdist will be larger for paired chain sequences.
tt = TCRTransport(neighbor_radius=100, maximum_dist=400)

# Add the reference repertoire. The reference bool must be set to True.
tt.add_repertoire(FILE1, beta_cols=['cdr3b', 'vb'], alpha_cols=['cdr3a', 'va'], reference=True)

# Add the repertoire which is to be compared to the reference repertoire and in which we will look for outlier sequences and clusters.
tt.add_repertoire(FILE2, beta_cols=['cdr3b', 'vb'], alpha_cols=['cdr3a', 'va'])
```

The rest of the functions are as in the single chain case.


## References
- Olson, B.J., Schattgen, S.A., Thomas, P.G., Bradley, P. and Matsen IV, F.A., 2022. Comparing T cell receptor repertoires using optimal transport. _PLOS Computational Biology_, 18(12), p.e1010681. [https://doi.org/10.1371/journal.pcbi.1010681](https://doi.org/10.1371/journal.pcbi.1010681)