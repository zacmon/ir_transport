# IRTransport
IRTransport uses optimal transport with TCRdist as the metric to detect amino acid sequences which are enriched in one dataset compared to another nonparametrically.
IRTransport can also use bespoke distance and neighbor-finding functions provided by the user.

This is a refactoring and updating of [transport](https://github.com/matsengrp/transport/tree/main) from the Matsen lab.
IRTransport also includes a SegmentedLinearModel class which closely resembles Vito Muggeo's [segmented R](https://cran.r-project.org/web/packages/segmented/index.html) package module for inferring breakpoints using a linear model.

## Installation
1. Clone the repository.

```bash
git clone https://github.com/zacmon/ir_transport.git
```

2. Pip install the package
```bash
pip install .
```

## Usage
Consider two TCR sequence datasets stored at `FILE1` and `FILE2`, where the first dataset is to be used as the reference.
I.e., we want to see which sequences from the second dataset are distinct from the first.
Assume that there are the columns `'cdr3b', 'vb'` which point to the BCDR3 and TRBV gene for the TRB receptor sequence.

```python
from ir_transport import IRTransport

# Create the IRTransport class object.
irt = IRTransport()

# Add the reference repertoire. The reference bool must be set to True.
irt.add_dataset(FILE1, seq_cols=['cdr3b', 'vb'], reference=True)

# Add the repertoire which is to be compared to the reference repertoire and in
# which we will look for outlier sequences and clusters.
irt.add_dataset(FILE2, seq_cols=['cdr3b', 'vb'])

# Compute the sample enrichment on the sample sequences and the reference sequences.
irt.compute_enrichment()

# Compute the p values of the observed enrichment scores on both the sample and
# reference datasets using a randomization test.
irt.compute_significance()

# Create clusters of TCRs using the enrichment score to identify the focal sequence
# and TCRdist to obtain neighbors.
irt.create_clusters()
```

It is also possible to compute statistics on both the sample and reference datasets in the same IRTransport object.

```python
from ir_transport import IRTransport

# Create the IRTransport class object.
irt = IRTransport()

# Add the reference repertoire. The reference bool must be set to True.
irt.add_dataset(FILE1, seq_cols=['cdr3b', 'vb'], reference=True)

# Add the repertoire which is to be compared to the reference repertoire and in
# which we will look for outlier sequences and clusters.
irt.add_dataset(FILE2, seq_cols=['cdr3b', 'vb'])

# Compute the sample enrichment on the sample sequences and the reference sequences.
irt.compute_enrichment(compute_reference_enrichment=True)

# Compute the p values of the observed enrichment scores on both the sample and
# reference datasets using a randomization test.
irt.compute_significance(compute_reference_significance=True)

# Create clusters of TCRs using the enrichment score to identify the focal sequence
# and TCRdist to obtain neighbors.
irt.create_clusters()
irt.create_clusters(dataset='reference')
```

If paired-chain data is available, you can point to the columns containg the TRAV gene and ACDR3 sequence. Here we assume, they are pointed to by `'cdr3a', 'va'` (of course, this will depend on the format of your data).

```python
from ir_transport import IRTransport

# Create the TCRTransport class object. We increase the neighbor_radius from the
# default of 48 since TCRdist will be larger for paired chain sequences.
irt = IRTransport(neighbor_radius=100, maximum_dist=400)

# Add the reference repertoire. The reference bool must be set to True.
irt.add_dataset(FILE1, seq_cols=['cdr3b', 'vb', 'cdr3a', 'va'], reference=True)

# Add the repertoire which is to be compared to the reference repertoire and
# in which we will look for outlier sequences and clusters.
tt.add_dataset(FILE2, seq_cols=['cdr3b', 'vb', 'cdr3a', 'va'])
```

The rest of the functions are as in the single chain case.

## References
- Olson, B.J., Schattgen, S.A., Thomas, P.G., Bradley, P. and Matsen IV, F.A. (2022). Comparing T cell receptor repertoires using optimal transport. _PLOS Computational Biology_, 18(12), p.e1010681. [https://doi.org/10.1371/journal.pcbi.1010681](https://doi.org/10.1371/journal.pcbi.1010681)
- Muggeo V.M.R. (2003). segmented: Regression Models with Break-Points / Change-Points Estimation (with Possibly Random Effects). [https://doi.org/10.32614/CRAN.package.segmented](https://doi.org/10.32614/CRAN.package.segmented)
