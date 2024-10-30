from functools import partial
import logging
from typing import *

import numpy as np
from numpy.typing import NDArray
import ot
import polars as pl
from tqdm import tqdm

from ir_transport.adjust_pvalues import get_adjusted_pvalues
from ir_transport.segmented_linear_model import SegmentedLinearModel
from ir_transport.utils import *

logging.getLogger().setLevel(logging.INFO)

SUPPORTED_SPECIES = {'human'}

def get_mass_distribution(
    df: pl.DataFrame,
    cols: Sequence[str],
    distribution_type: str = 'uniform',
) -> NDArray[np.float64]:
    """
    Obtain the mass distribution for the optimal transport algorithm.

    Parameters
    ----------
    df : polars.DataFrame
        A DataFrame containing sequences. It is expected that this DataFrame
        was produced by the utils.load_data function. It must have a
        'multiplicity' column.
    cols : sequence of str
        The columns used to define distinct sequences (over multiple columns).
    distribution_type : str, default 'uniform'
        How the mass distribution is contructions.
        'uniform': each unique clone is weighted by its multiplicity.

    Returns
    -------
    numpy.ndarray of numpy.float64
        The mass distirbution of the clones in the DataFrame.
    """
    if distribution_type == 'inverse_to_v_gene':
        pass
    elif distribution_type == 'uniform':
        return df.with_columns(
            clone=pl.concat_str(cols)
        ).with_columns(
            mass=(
                pl.col('multiplicity').sum().over('clone')
                / pl.col('multiplicity').sum()
            )
        )['mass'].to_numpy()
    else:
        raise RuntimeError(
            'Unsupported distribution_type. Valid values for distribution_type '
            'are \'inverse_to_v_gene\' and \'uniform\'.'
        )

def compute_effort_matrix(
    mass_1: NDArray[np.float64],
    mass_2: NDArray[np.float64],
    distance_matrix: NDArray[np.float64],
    reg: float = 0.01,
    **kwargs: Dict[str, Any],
) -> NDArray[np.float64]:
    """
    Infer the optimal transport map and compute the effort matrix.

    Parameters
    ----------
    mass_1 : numpy.ndarray of numpy.float64
        A mass distribution whose domain is ordered with the rows of the distance
        matrix.
    mass_2 : numpy.ndarray of numpy.float64
        A mass distribution whose doamin is ordered with the columns of the
        distance matrix.
    distance_matrix : numpy.ndarray of float64
        A matrix of distances computed between all-to-all comparisons of the
        domains of the two distributions.
    reg : float, default 0.01
        The regularization weight for the Sinkhorn solver.
    **kwargs
        Keyword arguments to ot.sinkhorn.

    Returns
    -------
    numpy.ndarray of numpy.float64
        The effort matrix, which is the elementwise product of the distance matrix
        and the optimal transport map.
    """
    ot_mat = ot.sinkhorn(mass_1, mass_2, distance_matrix, reg, **kwargs)
    return distance_matrix * ot_mat

def compute_enrichments(
    df: pl.DataFrame,
    effort_matrix: NDArray[np.float64],
    neighbor_map: pl.DataFrame,
    index_col: str = 'index',
    neighbor_index_col: str = 'n_index',
    max_distance: float = 200,
    axis: int = 0,
    no_neighbors: bool = False,
) -> pl.DataFrame:
    """
    Compute the loneliness for the entries in the DataFrame with or without neighbors.

    Parameters
    ----------
    df : polars.DataFrame
        A DataFrame containing the immune receptor sequences.
    effort_matrix : numpy.ndarray of numpy.float64
        The elementwise product of the optimal transport map and the distance
        matrix.
    neighbor_map : polars.DataFrame
        DataFrame containing the indices of the sequences with neighbors, the indices
        of the neighbors, and the distances between the two.
    index_col : str, default 'index'
        The column common to df and neighbor_map used to join the two for adding in
        neighbor enrichments and counts.
    max_distance : float, default 200
        The maximum distance which was used for scaling when inferring the optimal
        transport map.
    axis : int, default 0
        The axis along which the effort matrix will be summed to compute a sequence's
        loneliness. If axis = 0, then the effort matrix along axis 1 should have the
        same length as the given DataFrame and vice versa if axis = 1.
    no_neighbors : bool, default False
        The enrichment score will be each sequence's loneliness without adding
        contributions from neighbors.

    Returns
    -------
    df : polars.DataFrame
        The input DataFrame with columns 'enrichment' and 'num_neighbors', giving
        the enrichment score and the number of neighbors, respectively, for that
        sequence.
    """
    efforts = max_distance * df['multiplicity'].sum() * effort_matrix.sum(axis)

    df = df.with_columns(
        enrichment=efforts,
        num_neighbors=pl.lit(0, dtype=pl.UInt32)
    )

    if no_neighbors or len(neighbor_map) == 0:
        return df

    neighbor_map = neighbor_map.with_columns(
        neighbor_enrichment=efforts[neighbor_map[neighbor_index_col]]
    ).group_by(
        index_col
    ).agg(
        tmp_enrichment=pl.col('neighbor_enrichment').sum(),
        tmp_num_neighbors=pl.col('neighbor_enrichment').len()
    )

    df = df.join(
        neighbor_map,
        on=index_col,
        how='left'
    ).fill_null(
        0
    ).with_columns(
        enrichment=pl.col('enrichment') + pl.col('tmp_enrichment'),
        num_neighbors=pl.col('num_neighbors') + pl.col('tmp_num_neighbors')
    ).drop(
        '^tmp.*$'
    ).cast({
        'num_neighbors': pl.UInt32
    })

    return df

def split_datasets(
    df: pl.DataFrame,
    n: int,
    seed: int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence = None,
) -> Tuple[pl.DataFrame]:
    """
    Split a DataFrame by sampling without replacement.

    Parameters
    ----------
    df : polars.DataFrame
        A DataFrame.
    n : int
        The size of one partition of the DataFrame.
    seed : int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        The seed for the random number generator.

    Returns
    -------
    tuple of polars.DataFrame
        Two DataFrames that were obtained by sampling the given DataFrame without
        replacement.
    """
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    permutation = rng.permutation(df.shape[0])
    return  df[permutation[:n]], df[permutation[n:]]

class IRTransport():
    def __init__(
        self,
        species: str = 'human',
        distribution_type: str = 'uniform',
        lambd: float = 0.01,
        max_distance: float = 200,
        neighbor_radius: int = 48,
        distance_func: str | Callable = 'tcrdist',
        neighbor_func: Optional[Callable] = None,
        seed: int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence = None,
    ) -> None:
        """
        Initialize a IRTransport object.

        Parameters
        ----------
        species : str, default 'human'
            This specifies which database should be used when computing TCRdist
            among sequences.
            Presently the only acceptable option is 'human'.
        distribution_type : str, default 'uniform'
            How the mass distributions of a dataset should be computed.
        lambd : float, 0.01
            The regularization weight for the Sinkhorn solver.
        max_distance : float, default 200
            The maximum distance for inferring clusters of enriched sequences.
            Additionally, this value is used to scale the distance matrix when
            inferring an optimal transport map.
        neighbor_radius : int, default 48
            The inclusive distance at which a sequence is considered another sequence's
            neighbor when computing enrichments.
        distance_func : str or callable, default 'tcrdist'
            If string, this gives the distance metric used by tcrdist_rs.
            If a function, this must follow the utils.compute_distances format.
        neighbor_func : callable, optional
            If None, utils.get_neighbor_map is used. If not None, this should
            follow the utils.get_neighbor_map function and use the same distance
            as that given by distance_func.
        seed : int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            The seed for the random number generator. This is used when computing
            the significance of enrichments.
        """
        if species not in SUPPORTED_SPECIES:
            supported_species_str = f'{SUPPORTED_SPECIES}'[1:-1]
            raise ValueError(
                f'{species} is not a valid option for species. species must be '
                f'one of {supported_species_str}.'
            )
        self.distribution_type = distribution_type
        self.lambd = lambd
        self.max_distance = max_distance
        self.neighbor_radius = neighbor_radius
        self.rng = np.random.default_rng(seed)

        if isinstance(distance_func, str):
            self.distance_func = partial(
                compute_distances, distance=distance_func, species=species
            )
            self.neighbor_func = partial(
                get_neighbor_map, distance=distance_func, species=species
            )
        elif neighbor_func is None:
            raise RuntimeError(
                'If a custom distance_func is provided, a custom neighbor_func '
                'must be provided.'
            )
        else:
            self.distance_func = distance_func
            self.neighbor_func = neighbor_func

    def add_dataset(
        self,
        data: str | pl.DataFrame,
        seq_cols: str | Sequence[str],
        reference: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Preprocess a dataset and specify whether if it used as a reference.

        Parameters
        ----------
        data : str or dict of {str : any} or Sequence or numpy.ndarray or polars.Series or pandas.DataFrame or polars.DataFrame
            A variable which points to the dataset. If a string, data must
            point to a valid file to be read. Otherwise, data could be an existing
            DataFrame or two-dimensional data in many forms.
        seq_cols : str or sequence of str
            The column(s) pointing to the sequences to be compared.
            If seq_cols is a string or list containing one string, tcrdist is used.
            If seq_cols contains two strings, it is assumed that seq_cols points
            to a CDR3 and V gene, in that order. If seq_cols contains three strings,
            it is assumed that seq_cols points to the CDR1, CDR2, and CDR3, in that order.
            If seq_cols contains four strings, it is assumed that the CDR3, V gene,
            CDR3, V gene of the heavy/beta chain and light/alpha chain. If seq_cols
            contains six strings, it is assumed that it points to the heavy CDR1, CDR2,
            and CDR3 and light CDR1, CDR2, and CDR3, in that order.
        reference : bool, default False
            Specifies if the dataset being added will be used as the reference,
            i.e., its sequences will not be checked to see if they're enriched.
        **kwargs
            Keyword arguments to utils.load_data.

        Returns
        -------
        None
        """
        self.seq_cols = seq_cols

        if isinstance(seq_cols, str):
            self.seq_cols = seq_cols
            self.common_cols = [self.seq_cols,] + ['multiplicity']
            v_cols = None
        else:
            self.common_cols = self.seq_cols + ['multiplicity']
            if len(seq_cols) == 2:
                v_cols = seq_cols[1:]
                seq_cols = seq_cols[:1]
            elif len(seq_cols) == 4:
                v_cols = seq_cols[1::2]
                seq_cols = seq_cols[::2]

        df = load_data(data, seq_cols, v_cols, **kwargs).sort(
            # Sort for consistency between runs.
            self.seq_cols
        ).with_row_index()

        if hasattr(self.distance_func, 'keywords'):
            if 'tcrdist' in self.distance_func.keywords.values():
                only_nt = df.select(
                    res=pl.any_horizontal(
                        pl.col(seq_cols).str.contains('^[TGAC~_\*]+$')
                    ).any()
                )['res'].item(0)
                if only_nt:
                    raise RuntimeError(
                        'tcrdist is being used as the distance metric, but at '
                        'least one column for defining sequences contains '
                        'nucleotide sequences only. Column(s) for defining the '
                        f'sequences: {seq_cols}.'
                    )

        if reference:
            self.df_ref = df
        else:
            self.df_samp = df

    def compute_enrichment(
        self,
        no_neighbors: bool = False,
        compute_reference_enrichment: bool = False,
        **kwargs: Dict[str, Any]
    ) -> pl.DataFrame | Tuple[pl.DataFrame]:
        """
        Compute the enrichments of sequences in the non-reference dataset.

        The distances among sequences in the sample are computed, the mass
        distirbutions of each sample are computed, and the distance matrix
        among all sequences in the reference and sample datasets are computed.
        Then the effort matrix is computed by inferring an optimal transport map.
        Finally, the enrichments are computed.

        Parameters
        ----------
        no_neighbors : bool, default False
            If True, do not incorporate neighbors when computing enrichments.
        compute_reference_enrichment : bool, default False
            If True, compute the enrichments of the sequences in the reference dataset.
        **kwargs
            Keyword arguments to ot.sinkhorn.

        Returns
        -------
        self.df_samp : pl.DataFrame
            The sample DataFrame with columns containing information about the
            enrichment score and the number of neighbors the sequence has.
        self.df_ref : pl.DataFrame
            The reference DataFrame with columns containing information about the
            enrichment score and the number of neighbors the sequence has.
            This is returned in compute_reference_enrichment=True.
        """
        self.sample_neighbor_map = self.neighbor_func(
            self.df_samp[self.seq_cols].to_numpy(), self.neighbor_radius,
        )
        self.mass_ref = get_mass_distribution(self.df_ref, self.seq_cols)
        self.mass_samp = get_mass_distribution(self.df_samp, self.seq_cols)
        self.distance_matrix = self.distance_func(
            self.df_ref[self.seq_cols].to_numpy(),
            self.df_samp[self.seq_cols].to_numpy(),
        ) / self.max_distance

        self.effort_matrix = compute_effort_matrix(
            self.mass_ref, self.mass_samp, self.distance_matrix, self.lambd, **kwargs
        )

        self.df_samp = compute_enrichments(
            self.df_samp, self.effort_matrix, self.sample_neighbor_map,
            max_distance=self.max_distance, no_neighbors=no_neighbors
        )

        if compute_reference_enrichment:
            self.reference_neighbor_map = self.neighbor_func(
                self.df_ref[self.seq_cols].to_numpy(), self.neighbor_radius
            )
            self.df_ref = compute_enrichments(
                self.df_ref, self.effort_matrix, self.reference_neighbor_map,
                max_distance=self.max_distance, no_neighbors=no_neighbors,
                axis=1
            )
            return self.df_samp, self.df_ref

        return self.df_samp

    def compute_significance(
        self,
        trial_count: int = 100,
        trial_type: str = 'total',
        seed: int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence = None,
        compute_reference_significance: bool = False
    ) -> pl.DataFrame | Tuple[pl.DataFrame]:
        """
        Compute the significance of the calculated enrichment using a randomization test.

        In a single randomization trial, the input datasets are concatenated
        and randomly partitioned into the original sizes of the datasets.
        Enrichments are calculated for sequences in the shuffled dataset
        which is the same size as the original sample dataset, and p values
        are estimated using ECDFS obtained from the trials.

        Paramaters
        ----------
        trial_count : int, default 100
            The number of times the data will be randomized in order to compute
            significance.
        trial_type : str, 'total'
            trial_type = 'total' specifies that trial_count is the number of
            randomizations performed. trial_type = 'target' specifies that
            randomizations will be performed until the minimum number of times
            a sequence is seen in a dataset is equal to trial_count.
        seed : int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            The seed for the random number generator. If None, the object's rng
            attribute will be used.
        compute_reference_significance : bool, default False
            Compute the significance of the sequences in the reference dataset
            simultaneously.

        Returns
        -------
        self.df_samp : polars.DataFrame
            The sample dataset  DataFrame annotated with statistics from the
            randomization test.
        self.df_ref : polars.DataFrame
            The reference dataset  DataFrame annotated with statistics from the
            randomization test. This is returned if compute_reference_significance=True.
        """
        if 'enrichment' not in self.df_samp.columns:
            raise RuntimeError(
                'No enrichment values were computed for the original sample '
                'dataset, so significance cannot be computed. Please run '
                'compute_enrichment() using the IRTransport object '
                'and then try running this function again.'
            )

        if compute_reference_significance:
            if 'enrichment' not in self.df_ref.columns:
                raise RuntimeError(
                    'No enrichment values were computed for the original reference '
                    'dataset, so significance cannot be computed. Please run '
                    'compute_enrichment(compute_reference_enrichment=True) using '
                    'the IRTransport object and then try running this function again.'
                )

        if trial_type != 'target' and trial_type != 'total':
            raise ValueError('trial_type must be \'total\' or \'target\'.')

        if seed is not None:
            rng = self.rng
        else:
            rng = np.random.default_rng(seed)

        # TODO Change to weighted sampling instead of expanding fully?
        # TODO Avoid concatenating both DataFrames and pull samples directly
        #      from self.df_samp and self.df_rep?
        df_full = pl.concat((
            # Expand the DataFrame to the full size of the datasets to sample
            # sequences individually, labeling which sequences came from which DataFrame.
            df.select(
                self.common_cols
            ).with_row_index().group_by(
                self.common_cols
            ).agg(
                sample=func(pl.col('multiplicity').sum(), dtype=pl.Int8),
                index=pl.col('index').first()
            ).explode(
                'sample'
            )
            for df, func in zip((self.df_ref, self.df_samp), (pl.zeros, pl.ones))
        ))

        num_ref = self.df_ref['multiplicity'].sum()

        sample_randomized_scores = []
        reference_randomized_scores = []

        if trial_type == 'target':
            # Create a DataFrame that will be used to keep track of how many times
            # sequences from the sample dataset are seen.
            samp_seen = self.df_samp.select(
                self.seq_cols
            ).with_columns(
                num_seen=pl.lit(0)
            )

            if compute_reference_significance:
                # Create a DataFrame that will be used to keep track of how many
                # times sequences from the reference dataset are seen.
                ref_seen = self.df_ref.select(
                    self.seq_cols
                ).with_columns(
                    num_seen=pl.lit(0)
                )

        record_sequences_seen = lambda x, y: x.with_columns(
            num_seen=pl.when(
                pl.struct(self.seq_cols).is_in(y[self.seq_cols])
            ).then(
                pl.col('num_seen') + 1
            ).otherwise(
                pl.col('num_seen')
            )
        )

        counter = 0
        desc = 'Randomizing datasets and computing enrichment'
        with tqdm(desc=desc, position=0, total=trial_count) as progress_bar:
            while counter < trial_count:
                df_ref_trial, df_samp_trial = split_datasets(df_full, num_ref, rng)

                df_ref_trial = df_ref_trial.group_by(
                    self.seq_cols
                ).agg(
                    multiplicity=pl.len()
                ).with_row_index()

                df_samp_trial = df_samp_trial.group_by(
                    self.seq_cols
                ).agg(
                    multiplicity=pl.len()
                ).with_row_index()

                mass_ref_trial = get_mass_distribution(df_ref_trial, self.seq_cols)
                mass_samp_trial = get_mass_distribution(df_samp_trial, self.seq_cols)

                samp_trial_neighbor_map = self.neighbor_func(
                    df_samp_trial[self.seq_cols].to_numpy(), self.neighbor_radius,
                )

                distance_matrix = self.distance_func(
                    df_ref_trial[self.seq_cols].to_numpy(),
                    df_samp_trial[self.seq_cols].to_numpy(),
                ) / self.max_distance

                effort_matrix = compute_effort_matrix(
                    mass_ref_trial, mass_samp_trial, distance_matrix, self.lambd
                )

                df_samp_trial = compute_enrichments(
                    df_samp_trial, effort_matrix, samp_trial_neighbor_map,
                    max_distance=self.max_distance,
                )

                # Keep only those sequences which appeared in the true sample dataset.
                samp_trial_joined = self.df_samp.join(
                    df_samp_trial,
                    on=self.seq_cols,
                    suffix='_trial',
                )

                sample_randomized_scores.append(
                    samp_trial_joined
                )

                if not compute_reference_significance:
                    if trial_type == 'total':
                        counter += 1
                    else:
                        samp_seen = record_sequences_seen(samp_seen, samp_trial_joined)
                        counter = samp_seen['num_seen'].min()

                    if progress_bar.n != counter:
                        progress_bar.update(1)

                    continue

                ref_trial_neighbor_map = self.neighbor_func(
                    df_ref_trial[self.seq_cols].to_numpy(), self.neighbor_radius
                )
                df_ref_trial = compute_enrichments(
                    df_ref_trial, effort_matrix, ref_trial_neighbor_map,
                    max_distance=self.max_distance, axis=1
                )
                ref_trial_joined = self.df_ref.join(
                    df_ref_trial,
                    on=self.seq_cols,
                    suffix='_trial'
                )
                reference_randomized_scores.append(
                    ref_trial_joined
                )

                if trial_type == 'total':
                    counter += 1
                else:
                    samp_seen = record_sequences_seen(samp_seen, samp_trial_joined)
                    ref_seen = record_sequences_seen(ref_seen, ref_trial_joined)
                    counter = min(samp_seen['num_seen'].min(), ref_seen['num_seen'].min())

                if progress_bar.n != counter:
                    progress_bar.update(1)

        df_samp_rand = pl.concat(sample_randomized_scores)
        num_sample_seqs_seen = df_samp_rand.unique(self.seq_cols).shape[0]
        if num_sample_seqs_seen != self.df_samp.shape[0]:
            num_missing = self.df_samp.shape[0] - num_sample_seqs_seen
            raise RuntimeError(
                'Not all sequences were seen from the sample dataset over all '
                f'trials (missing {num_missing} from {self.df_samp.shape[0]} total). '
                'Increase the trial_count.'
            )

        min_sample_size = df_samp_rand.select(
            pl.len().over(self.seq_cols).min()
        )['len'].item(0)

        if min_sample_size == 1:
            raise RuntimeError(
                'The minimum number of times a sample-dataset sequence was seen '
                'was 1. This will result in poor statistics. Rerun with an increased '
                'trial_count.'
            )

        if compute_reference_significance:
            df_ref_rand = pl.concat(reference_randomized_scores)
            num_reference_seqs_seen = df_ref_rand.unique(self.seq_cols).shape[0]
            if num_reference_seqs_seen != self.df_ref.shape[0]:
                num_missing = self.df_ref.shape[0] - num_sample_seqs_seen
                raise RuntimeError(
                    'Not all sequences were seen from the reference dataset over all '
                    f'trials (missing {num_missing} from {self.df_ref.shape[0]} total).'
                    'Increase the trial_count.'
                )

            min_ref_size = df_ref_rand.select(
                pl.len().over(self.seq_cols).min()
            )['len'].item(0)

            if min_ref_size == 1:
                raise RuntimeError(
                    'The minimum number of times a reference-dataset sequence was seen '
                    'was 1. This will result in poor statistics. Rerun with an increased '
                    'trial_count.'
                )

            min_sample_size = min(min_sample_size, min_ref_size)

        estimate_significance = lambda x: x.with_columns(
            idx=pl.int_range(pl.len()).over(self.seq_cols)
        ).filter(
            # Downsample all sequences to the same amount.
            pl.col('idx') < min_sample_size
        ).group_by(
            pl.exclude('^*trial$', 'idx')
        ).agg(
            scores=pl.col('enrichment_trial').sort(),
            z_score=(
                (pl.col('enrichment').first() - pl.col('enrichment_trial').mean())
                / pl.col('enrichment_trial').std()
            ),
            ecdf=pl.int_range(0, pl.len() + 1) / pl.len(),
            search_sort=pl.col('enrichment_trial').sort().search_sorted(pl.col('enrichment').first())
        ).with_columns(
            p_value=1 - pl.col('ecdf').list.get(pl.col('search_sort'))
        ).drop(
            ['ecdf', 'search_sort']
        ).sort(
            'index'
        )

        self.df_samp = estimate_significance(df_samp_rand)

        if not compute_reference_significance:
            return self.df_samp

        self.df_ref = estimate_significance(df_ref_rand)

        return self.df_samp, self.df_ref

    def adjust_pvalues(
        self,
        method: str = 'storey',
        **kwargs: Dict[str, Any]
    ) -> pl.DataFrame | Tuple[pl.DataFrame]:
        """
        Adjust p values for multiple testing.

        If p values for the sample and reference datasets are both computed,
        their p values are corrected together.

        The default method used for correction ('storey') gives Storey q values.
        Ideally, the histogram of p values is inspected prior to applying
        multiple testing corrections. If the histogram of p values is flat
        for p > 0.5, then Storey q values are appropriate. If the histogram
        of p values is noisy or the size of the dataset is small,
        Benjamini-Hochberg ('bh') might be more appropriate.

        Parameters
        ----------
        method : str, default 'storey'
            Method using for adjusting the p values. Available methods:
                'bonferonni' _[1]
                'sidak' _[2]
                'empirical_null'
                'holm' _[3]
                'hommel' _[4]
                'simes-hochberg' _[5] _[6]
                'bh' _[7]
                'by' _[8]
                'storey' _[9] _[10]
        **kwargs : dict of { str : any }
            Keyword arguments to ir_transport.get_adjusted_pvalues.

        Returns
        -------
        self.df_samp : polars.DataFrame
            The sample dataset  DataFrame annotated with adjusted p values.
        self.df_ref : polars.DataFrame
            The reference dataset  DataFrame annotated with adjusted p values.

        References
        ----------
        .. [1] Neyman J, Pearson ES (1928) "On the use and interpretation of certain
               test criteria for purposes of statistical inference: Part I." Biometrika
               20A(1/2): 175-240. https://doi.org/10.2307/2331945
        .. [2] Sidak Z (1967) "Rectangular confidence regions for the means of
               multivariate normal distributions." J Am Stat Assoc 62: 626-633.
               https://doi.org/10.2307/2283989
        .. [3] Holm S (1979) "A Simple Sequentially Rejective Multiple Test Procedure."
               Scand Stat Theory Appl 6(2): 65-70. https://www.jstor.org/stable/4615733
        .. [4] Hommel G (1988) "A stagewise rejective multiple test procedure based on
               a modified Bonferroni test." Biometrika 75(2): 383-386.
               https://doi.org/10.1093/biomet/75.2.383
        .. [5] Simes RJ (1986) "An improved Bonferroni procedure for multiple tests of
               significance." Biometrika 73(3): 751-754.
               https://doi.org/10.1093/biomet/73.3.751
        .. [6] Hochberg Y (1988) "A sharper Bonferroni procedure for multiple tests of
               significance." Biometrika 75(4): 800-802.
               https://doi.org/10.1093/biomet/75.4.800
        .. [7] Hochberg Y, Benjamini Y (1990) "More powerful procedures for multiple
                significance testing." Stat Med 9(7): 811-818.
                https://doi.org/10.1002/sim.4780090710
        .. [8] Benjamini Y, Yekutieli D (2001) "The control of the false discovery
                rate in multiple testing under dependency." Ann Statist 29(4): 1165-1188.
                https://doi.org/10.1214/aos/1013699998
        .. [9] Storey JD, Tibshirani R. (2003) "Statistical significance for
                genomewide studies." Proc Natl Acad Sci U S A. 100(16):9440-5.
                https://doi.org/10.1073/pnas.1530509100
        .. [10] Storey, JD et al. (2004) "Strong control, conservative point estimation
                and simultaneous conservative consistency of false discovery rates: a
                unified approach." J. R. Stat. Soc., B: Stat. 66: 187-205.
                https://doi.org/10.1111/j.1467-9868.2004.00439.x
        """
        if 'p_value' not in self.df_samp.columns:
            raise RuntimeError(
                'p values cannot be adjusted since no p values have been computed. '
                'Please run compute_significance() using the IRTransport object '
                'and then try running this function again.'
            )

        if 'p_value' not in self.df_ref.columns:
            p = self.df_samp['p_value']
            p_adj = get_adjusted_pvalues(p, method, **kwargs)
            self.df_samp = self.df_samp.with_columns(
                **{f'p_value_{method}': p_adj}
            )
        else:
            p = pl.concat((
                self.df_samp['p_value'], self.df_ref['p_value']
            ))
            p_adj = get_adjusted_pvalues(p, method, **kwargs)
            len_df_samp = len(self.df_samp)

            self.df_samp = self.df_samp.with_columns(
                **{f'p_value_{method}': p_adj[:len_df_samp]}
            )
            self.df_ref = self.df_ref.with_columns(
                **{f'p_value_{method}': p_adj[len_df_samp:]}
            )

    def cluster(
        self,
        df: Optional[pl.DataFrame] = None,
        step: int = 5,
        init_breakpoint: float = None,
        quantile: float = 0.5,
        return_intermediates: bool = False,
        debug: bool = False,
        **kwargs: Dict[str, Any],
    ) -> pl.DataFrame | Tuple[pl.DataFrame, SegmentedLinearModel]:
        """
        Obtain the cluster of sequences around the most enriched sequence using
        segmented linear regression.

        Parameters
        ----------
        df : polars.DataFrame, optional
            A DataFrame containing sequences and their enrichments.
        step : int, default 5
            The width of the annuluses around the most enriched sequence.
        init_breakpoint: int, default optional
            The initial guess at which there is a breakpoint in the mean enrichment
            in an annulus vs. annulus radius.
        quantile : float, default 0.5
            Sequences with enrichments at least as large as the enrichment specified
            by this quantile will be considered as potential neighbors
            to the clusters.
        return_intermediates : bool, default False
            Return the DataFrame of the cluster around the most enriched sequence
            as well as the DataFrame containing the annulus enrichments and the
            SegmentedLinearModel object used to infer the breakpoint.
        debug : bool, default False
            Return the DataFrame containing the annulus enrichments and the
            SegmentedLinearModel object if a segmented linear regression
            doesn't fit the data well.
        **kwargs : dict of { str : any }
            Keyword arguments to SegmentedLinearModel.fit().

        Returns
        -------
        df : polars.DataFrame
            The DataFrame containing the sequences which clustered around
            the most enriched sequence present in df.
        tmp : polars.DataFrame
            The DataFrame containing the measurements of the mean annulus enrichment
            for each annulus as well as the number of sequences in each annulus.
            return_intermediates = True will always return this. debug = True will
            return this if the segmented linear model is a poor fit.
        slm : SegmentedLinearModel
            The SegmentedLinearModel object used to fit the breakpoint, giving
            the radius of a cluster. return_intermediates = True will always
            return this. debug = True will return this if the segmented linear
            model is a poor fit.
        """
        if df is None:
            df = self.df_samp

        if 'enrichment' not in df.columns:
            raise RuntimeError(
                'No enrichment values were computed for the given input'
                'dataset\'s DataFrame, so a cluster cannot be create. Please '
                'compute the enrichment for the clones and then try running this '
                'function again.'
            )

        if quantile < 0 or quantile >= 1:
            raise ValueError('quantile must be in [0, 1).')

        max_score_arg_max = df['enrichment'].arg_max()
        max_score = df[max_score_arg_max, 'enrichment']

        # Compute the distance between the most enriched sequence in the DataFrame
        # and every sequence in the DataFrame.
        distance_vec = self.distance_func(
            df[max_score_arg_max, self.seq_cols].to_numpy(),
            df[self.seq_cols].to_numpy()
        ).ravel()

        radii = np.arange(0, self.max_distance + step, step)
        # Determine which sequences belong to which annulus.
        annulus_searchsort = np.searchsorted(radii, distance_vec, 'left')

        df = df.with_columns(
            enrichment_above_quantile=pl.col('enrichment') >= pl.col('enrichment').quantile(quantile),
            dist_to_max_score=distance_vec,
            # Assign sequences to annuluses.
            annulus_idx=annulus_searchsort
        ).filter(
            # Keep upper-quantile enrichment sequences only.
            (pl.col('enrichment_above_quantile'))
            # Remove sequences beyond the max annulus radius.
            & (pl.col('annulus_idx') < len(radii) - 1)
        ).with_columns(
            annulus_radius=pl.lit(radii).get(pl.col('annulus_idx'))
        )

        # Obtain the mean annulus enrichments for breakpoint inference.
        tmp = df.group_by(
            ['annulus_radius']
        ).agg(
            sum_enrichment=pl.col('enrichment').sum(),
            mean_annulus_enrichment=pl.col('enrichment').mean(),
            num_in_annulus=pl.len()
        ).sort(
            'annulus_radius'
        ).with_columns(
            num_in_cluster=pl.col('num_in_annulus').cum_sum(),
            mean_neighborhood_enrichment=(
                pl.col('sum_enrichment').cum_sum() / pl.col('num_in_annulus').cum_sum()
            )
        )

        slm = SegmentedLinearModel(
            tmp['annulus_radius'], tmp['mean_annulus_enrichment'],
        )
        try:
            slm.fit(init_breakpoint, **kwargs)
        except Exception as e:
            if 'Breakpoint analysis failed' in str(e):
                if debug:
                    return tmp, slm
                raise RuntimeError(
                    'Segmented linear model did not find a breakpoint.'
                )
            else:
                raise e

        regression_failed = False
        # No difference in slope.
        if slm.max0_params[0] < 1e-5:
            regression_failed = True
        # Breakpoint error is high.
        #elif slm.breakpoint_se[0] > slm.breakpoints[0]:
        #    regression_failed = True
        if regression_failed:
            if debug:
                return tmp, slm
            raise RuntimeError(
                'Segmented linear model did not find a breakpoint.'
            )

        cluster_radius = slm.breakpoints[0]

        df = df.filter(
            (pl.col('dist_to_max_score') <= cluster_radius )
        ).drop(
            ('annulus_idx', 'annulus_radius', 'enrichment_above_quantile', 'dist_to_max_score')
        )

        if return_intermediates:
            return df, tmp, slm
        return df

    def create_clusters(
        self,
        max_cluster_count: int = 10,
        step: int = 5,
        init_breakpoint: float = None,
        quantile: float = 0.5,
        dataset: str = 'sample',
        return_intermediates: bool = False,
        debug: bool = False,
        recompute_unclustered_enrichments: bool = False,
        **kwargs
    ) -> pl.DataFrame | Tuple[pl.DataFrame, List[pl.DataFrame], List[SegmentedLinearModel]]:
        """
        Create clusters around each consecutive unclustered but most enriched sequence
        in the sample or reference dataset.

        Parameters
        ----------
        max_cluster_count : int, default 10
            The number of clusters to form.
        step : int, default 5
            The width of the annuluses around the most enriched sequence. This
            is used to collect measurements and identify the breakpoint used
            to define a cluster.
        init_breakpoint: int, optional
            The initial guess at which there is a breakpoint in the mean enrichment
            in an annulus vs. annulus radius.
        quantile : float, default 0.5
            Sequences with enrichments at least as large as the enrichment specified
            by this quantile will which will be considered as potential neighbors
            to the clusters.
        dataset : str, default 'sample'
            Which dataset on which clustering will be performed.
        return_intermediates : bool, default False
            Return the DataFrame of the cluster around the most enriched sequence
            as well as the DataFrame containing the annulus enrichments and the
            SegmentedLinearModel object used to infer the breakpoint.
        debug : bool, default False
            Return the DataFrame containing the annulus enrichments and the
            SegmentedLinearModel object if a segmented linear regression
            doesn't fit the data well.
        recompute_unclustered_enrichments : bool, default False
            Compute the enrichments of the remaining unclustered dataset against
            the other dataset, and use those enrichments to perform clustering.
            The default option uses the enrichments computed from the full dataset.
        **kwargs : dict of { str : any }
            Keyword arguments to SegmentedLinearModel.fit().

        Returns
        -------
        df : polars.DataFrame
            The DataFrame containing the full dataset's sequences with an
            additional column 'transport_cluster' which annotates which cluster
            a sequence is in.
        tmp : list of polars.DataFrame
            List containing from each iteration the DataFrame of the measurements
            of the mean annulus enrichment for each annulus as well as the number
            of sequences in each annulus. return_intermediates = True will always
            return this. debug = True will return this if the segmented linear model
            is a poor fit.
        slm : SegmentedLinearModel
            List containing from each iteration the SegmentedLinearModel object
            used to fit the breakpoint, giving the radius of a cluster.
            return_intermediates = True will always return this. debug = True
            will return this if the segmented linear model is a poor fit.
        """
        if dataset == 'sample':
            df = self.df_samp
            msg = ''
        elif dataset == 'reference':
            df = self.df_ref
            msg = 'compute_reference_enrichment=True'
        else:
            raise ValueError(
                'dataset must be \'sample\' or \'reference\'.'
            )

        if 'enrichment' not in df:
            raise RuntimeError(
                f'No enrichment values were computed for the original {dataset} '
                'dataset, so no clusters can be created. Please run '
                f'compute_sample_enrichment({msg}) using the IRTransport object '
                'and then try running this function again.'
            )

        neighbor_map = getattr(self, f'{dataset}_neighbor_map')

        tmp_dfs = []
        slms = []

        progress_bar = tqdm(desc='Creating clusters', position=0, total=max_cluster_count)
        progress_bar_closed = False
        # Build the initial cluster from the full dataset.
        try:
            res = self.cluster(
                df, step, init_breakpoint, quantile, return_intermediates,
                debug, **kwargs
            )
            if debug:
                if isinstance(res, tuple) and len(res) == 2:
                    logging.info(
                        'Segmented linear model did not find a breakpoint. Returning '
                        'the DataFrame used to fit the model as well as the model object.'
                    )
                    return res
        except Exception as e:
            if 'did not find a breakpoint' in str(e):
                progress_bar.close()
                progress_bar_closed = True
                logging.info(f'{e} Terminating finding clusters.')
            else:
                raise e
            return
        else:
            if return_intermediates:
                df_cluster, tmp_df, slm = res
                tmp_dfs.append(tmp_df)
                slms.append(slm)
            else:
                df_cluster = res

            df_cluster = df_cluster.with_columns(
                transport_cluster=pl.lit(0)
            )

        if not progress_bar_closed:
            progress_bar.update(1)
        num_clusters = 1

        while num_clusters < max_cluster_count:
            # Obtain the subsample which excludes all the previously
            # clustered sequences.
            df_sub = df.filter(
                ~pl.col('index').is_in(df_cluster['index'])
            ).with_row_index(
                'sub_index'
            )

            if recompute_unclustered_enrichments:
                # Remove already clustered sequences from the neighbor map and map
                # the original indices to the indices in the subsample's DataFrame.
                sub_neighbor_map = neighbor_map.filter(
                    ~(pl.col('index').is_in(df_cluster['index']) | pl.col('n_index').is_in(df_cluster['index']))
                ).with_columns(
                    sub_index=pl.col('index').replace(
                        df_sub['index'],
                        df_sub['sub_index']
                    ),
                    n_index=pl.col('n_index').replace(
                        df_sub['index'],
                        df_sub['sub_index']
                    )
                )

                mass_sub = get_mass_distribution(df_sub, self.seq_cols)

                if dataset == 'sample':
                    # Get the subsample's distance matrix by using a view.
                    distance_matrix = self.distance_matrix[:, df_sub['index']]
                    effort_matrix = compute_effort_matrix(
                        self.mass_ref, mass_sub, distance_matrix, self.lambd
                    )
                    df_sub = compute_enrichments(
                        df_sub, effort_matrix, sub_neighbor_map, 'sub_index',
                        max_distance=self.max_distance
                    )
                else:
                    # Get the subsample's distance matrix by using a view.
                    distance_matrix = self.distance_matrix[df_sub['index'], :]
                    effort_matrix = compute_effort_matrix(
                        mass_sub, self.mass_samp, distance_matrix, self.lambd
                    )
                    df_sub = compute_enrichments(
                        df_sub, effort_matrix, sub_neighbor_map, 'sub_index',
                        max_distance=self.max_distance, axis=1
                    )

            df_sub = df_sub.drop('sub_index')

            try:
                res = self.cluster(
                    df_sub, step, init_breakpoint, quantile, return_intermediates,
                    debug, **kwargs
                )
                if debug:
                    if isinstance(res, tuple) and len(res) == 2:
                        logging.info(
                            'Segmented linear model did not find a breakpoint. Returning '
                            'the DataFrame used to fit the model as well as the model object.'
                        )
                        return res
            except Exception as e:
                if 'did not find a breakpoint' in str(e):
                    progress_bar.close()
                    progress_bar_closed = True
                    logging.info(f'{e} Terminating finding clusters.')
                    break
                else:
                    raise e
            else:
                if return_intermediates:
                    df_sub, tmp_df, slm = res
                    tmp_dfs.append(res[1])
                    slms.append(res[2])
                else:
                    df_sub = res
                df_cluster = pl.concat((
                    df_cluster,
                    df_sub.with_columns(
                        transport_cluster=pl.lit(num_clusters)
                    )
                ))

            num_clusters += 1
            progress_bar.update(1)

        if not progress_bar_closed:
            progress_bar.close()

        if dataset == 'sample':
            self.df_samp =  self.df_samp.join(
                df_cluster, on=self.common_cols, how='left'
            ).drop(
                pl.col('^*right$')
            )

            if return_intermediates:
                return self.df_samp, tmp_dfs, slms
            return self.df_samp
        else:
            self.df_ref =  self.df_ref.join(
                df_cluster, on=self.common_cols, how='left'
            ).drop(
                pl.col('^*right$')
            )

            if return_intermediates:
                return self.df_ref, tmp_dfs, slms
            return self.df_ref
