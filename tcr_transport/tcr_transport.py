import logging
from typing import *

import numpy as np
from numpy.typing import NDArray
import ot
import polars as pl

from tcr_transport.segmented_linear_model import SegmentedLinearModel
from tcr_transport.utils import *

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
        A DataFrame containing the immune receptor sequences. It is expected
        that this DataFrame was produced by the utils.load_data function.
        It must have a 'recomb_multiplicity' column.
    cols : sequence of str
        The columns used to define immune receptor clones.
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
                pl.col('recomb_multiplicity').sum().over('clone')
                / pl.col('recomb_multiplicity').sum()
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
    ot_mat = ot.sinkhorn(mass_1, mass_2, distance_matrix, reg)
    return distance_matrix * ot_mat

def compute_enrichments(
    df: pl.DataFrame,
    effort_matrix: NDArray[np.float64],
    distance_vectorform: NDArray[np.int16],
    max_distance: float = 200,
    neighbor_radius: int = 48,
    axis: int = 0,
    no_neighbors: bool = False,
):
    """
    Compute the loneliness for the entries in the DataFrame with or without neighbors.

    Parameters
    ----------
    df : polars.DataFrame
        A DataFrame containing the immune receptor sequences.
    effort_matrix : numpy.ndarray of numpy.float64
        The elementwise product of the optimal transport map and the distance
        matrix.
    distance_vectorform : numpy.ndarray of numpy.int16
        The distances among all sequences in the given DataFrame.
    max_distance : float, default 200
        The maximum distance which was used for scaling when inferring the optimal
        transport map.
    neighbor_radius : int, default 48
        The inclusive distance at which a sequence is considered another sequence's
        neighbor.
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
    efforts = max_distance * df['recomb_multiplicity'].sum() * effort_matrix.sum(axis)

    len_df = df.shape[0]

    df = df.with_columns(
        enrichment=efforts,
        num_neighbors=pl.lit(1)
    )

    if no_neighbors:
        return df

    neighbor_idxs = np.where(distance_vectorform <= neighbor_radius)
    if neighbor_idxs:
        # TODO Use a sparse representation and not a dense mask array?
        rows, cols = condensed_idx_to_square_indices(neighbor_idxs[0], len_df)
        mask = np.zeros((len_df,) * 2, dtype=bool)
        np.fill_diagonal(mask, True)
        mask[rows, cols] = mask[cols, rows] = True

        df = df.with_columns(
            enrichment=efforts @ mask,
            num_neighbors=mask.sum(0)
        )

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

class TCRTransport():
    def __init__(
        self,
        species: str,
        distribution_type: str = 'uniform',
        lambd: float = 0.01,
        max_distance: float = 200,
        neighbor_radius: int = 48,
        seed: int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence = None,
    ) -> None:
        """
        Initialize a TCRTransport object.

        Parameters
        ----------
        species : str
            This specifies which database should be used when computing TCRdist
            among sequences.
            Presently the only acceptable option is 'human'.
        distribution_type : str, default 'uniform'
            How the mass distributions of a repertoire should be computed.
        lambd : float, 0.01
            The regularization weight for the Sinkhorn solver.
        max_distance : float, default 200
            The maximum distance for inferring clusters of enriched sequences.
            Additionally, this value is used to scale the distance matrix when
            inferring an optimal transport map.
        neighbor_radius : int, default 48
            The inclusive distance at which a sequence is considered another sequence's
            neighbor when computing enrichments.
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
        self.species = species
        self.distribution_type = distribution_type
        self.lambd = lambd
        self.max_distance = max_distance
        self.neighbor_radius = neighbor_radius
        self.rng = np.random.default_rng(seed)

    def add_repertoire(
        self,
        data: str | pl.DataFrame,
        beta_cols: Optional[Sequence[str]] = None,
        alpha_cols: Optional[Sequence[str]] = None,
        reference: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Preprocess a repertoire and specify whether if it used as a reference.

        Parameters
        ----------
        data : str or dict of {str : any} or Sequence or numpy.ndarray or polars.Series or pandas.DataFrame or polars.DataFrame
            A variable which points to the repertoire data. If a string, data must
            point to a valid file to be read. Otherwise, data could be an existing
            DataFrame or two-dimensional data in many forms.
        beta_cols : sequence of str, optional
            The columns pointing to the CDR3 and V allele/gene of a VDJ sequence.
            The string pointing to the CDR3 must come first and the V allele/gene
            column must come second.
        alpha_cols : sequence of str, optional
            The columns pointing to the CDR3 and V allele/gene of a VJ sequence.
            The string pointing to the CDR3 must come first and the V allele/gene
            column must come second.
        reference : bool, default False
            Specifies if the repertoire being added will be used as the reference,
            i.e., its sequences will not be checked to see if they're enriched.
        **kwargs
            Keyword arguments to utils.load_data.

        Returns
        -------
        None
        """
        if beta_cols is None and alpha_cols is None:
            raise RuntimeError(
                'Both beta_cols and alpha_cols must not be None'
            )
        elif beta_cols is not None and alpha_cols is not None:
            seq_cols = beta_cols[:1] + alpha_cols[:1]
            v_cols = beta_cols[1:] + alpha_cols[1:]
            old_cols = beta_cols + alpha_cols
            self.tcr_cols = ['cdr3b', 'vb', 'cdr3a', 'va']
        elif beta_cols is None:
            seq_cols, v_cols = alpha_cols
            old_cols = alpha_cols
            self.tcr_cols = ['cdr3', 'v']
        else:
            seq_cols, v_cols = beta_cols
            old_cols = beta_cols
            self.tcr_cols = ['cdr3', 'v']

        df = load_data(data, seq_cols, v_cols, **kwargs).rename(
            {col: rcol for col, rcol in zip(old_cols, self.tcr_cols)}
        ).sort(
            # Sort for consistency between runs.
            self.tcr_cols
        ).with_row_index()

        if reference:
            self.df_ref = df
        else:
            self.df_samp = df

        self.common_cols = self.tcr_cols + ['recomb_multiplicity']

    def compute_sample_enrichment(
        self,
        no_neighbors: bool = False,
    ) -> None:
        """
        Compute the enrichments of sequences in the non-reference repertoire.

        The TCRdist among sequences in the sample are computed, the mass
        distirbutions of each sample are computed, and the TCRdist matrix
        among all sequences in the reference and sample repertoires are computed.
        Then the effort matrix is computed by inferring an optimal transport map.
        Finally, the enrichments are computed.

        Parameters
        ----------
        no_neighbors : bool, default False
            If True, do not incorporate neighbors when computing enrichments.

        Returns
        -------
        None
        """
        self.sample_distance_vectorform = compute_distance_vectorform(
            self.df_samp[self.tcr_cols].to_numpy()
        )
        self.mass_ref = get_mass_distribution(self.df_ref, self.tcr_cols)
        self.mass_samp = get_mass_distribution(self.df_samp, self.tcr_cols)
        self.distance_matrix = compute_distance_vectorform(
            self.df_ref[self.tcr_cols].to_numpy(),
            self.df_samp[self.tcr_cols].to_numpy(),
            dtype=np.float64
        ).reshape(self.df_ref.shape[0], self.df_samp.shape[0]) / self.max_distance

        self.effort_matrix = compute_effort_matrix(
            self.mass_ref, self.mass_samp, self.distance_matrix, self.lambd
        )

        self.df_samp = compute_enrichments(
            self.df_samp, self.effort_matrix, self.sample_distance_vectorform,
            self.max_distance, self.neighbor_radius, no_neighbors=no_neighbors
        )

    def compute_significance(
        self,
        trial_count: int = 100,
        seed: int | np.random.Generator | np.random.BitGenerator | np.random.SeedSequence = None,
    ) -> pl.DataFrame:
        """
        Compute the significance of the calculated enrichment using a randomization test.

        In a single randomization trial, the input repertoires are concatened
        and randomly partitioned into the original sizes of the repertoires.
        Enrichments are calculated for sequences in the shuffled repertoire
        which is the same size as the original sample repertoire, and p values
        are estimated.

        Paramaters
        ----------
        trial_count : int, default 100
            The number of times the data will be randomized in order to compute
            significance.
        seed : int or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
            The seed for the random number generator. If None, the object's rng
            attribute will be used.

        Returns
        -------
        df_rand : polars.DataFrame
            The sample repertoire DataFrame annotated with statistics from the
            randomization test.
        """
        if 'enrichment' not in self.df_samp.columns:
            raise RuntimeError(
                'No enrichment values were computed for the original sample '
                'repertoire, so no significance can be computed. Please run '
                'compute_sample_enrichment() using the TCRTransport object '
                'and then try running this function again.'
            )

        if seed is not None:
            rng = self.rng
        else:
            rng = np.random.default_rng(seed)

        #ref_distance_vectorform = compute_distance_vectorform(
        #    self.df_ref[self.tcr_cols].to_numpy()
        #)

        # TODO Change to weighted sampling instead of expanding fully?
        # TODO Avoid concatenating both DataFrames and pull samples directly
        #      from self.df_samp and self.df_rep?
        df_full = pl.concat((
            # Expand the DataFrame to the full size of the repertoires to sample
            # TCRs individually, labeling which TCRs came from which DataFrame.
            df.select(
                self.common_cols
            ).with_row_index().group_by(
                self.common_cols
            ).agg(
                sample=func(pl.col('recomb_multiplicity').sum(), dtype=pl.Int8),
                index=pl.col('index').first()
            ).explode(
                'sample'
            )
            for df, func in zip((self.df_ref, self.df_samp), (pl.zeros, pl.ones))
        ))

        num_ref = self.df_ref['recomb_multiplicity'].sum()

        randomized_scores = []

        for trial in range(trial_count):
            df_ref_trial, df_samp_trial = split_datasets(df_full, num_ref, rng)

            df_ref_trial = df_ref_trial.group_by(
                self.tcr_cols
            ).agg(
                recomb_multiplicity=pl.len()
            )

            df_samp_trial = df_samp_trial.group_by(
                self.tcr_cols
            ).agg(
                recomb_multiplicity=pl.len()
            ).with_row_index()

            mass_ref_trial = get_mass_distribution(df_ref_trial, self.tcr_cols)
            mass_samp_trial = get_mass_distribution(df_samp_trial, self.tcr_cols)

            # TODO Use precomputed distance matrix and vectorforms.
            samp_trial_dist_vectorform = compute_distance_vectorform(
                df_samp_trial[self.tcr_cols].to_numpy()
            )

            distance_matrix = compute_distance_vectorform(
                df_ref_trial[self.tcr_cols].to_numpy(),
                df_samp_trial[self.tcr_cols].to_numpy(),
                dtype=np.float64
            ).reshape(df_ref_trial.shape[0], df_samp_trial.shape[0]) / self.max_distance

            effort_matrix = compute_effort_matrix(
                mass_ref_trial, mass_samp_trial, distance_matrix, self.lambd
            )

            df_samp_trial = compute_enrichments(
                df_samp_trial, effort_matrix, samp_trial_dist_vectorform,
                self.max_distance, self.neighbor_radius
            )

            # Keep only those TCRs which appeared in the true second repertoire.
            randomized_scores.append(
                self.df_samp.join(
                    df_samp_trial,
                    on=self.common_cols,
                    suffix='_trial'
                )
            )

        df_rand = pl.concat(randomized_scores)

        num_tcrs_seen = df_rand.unique(self.common_cols).shape[0]
        if num_tcrs_seen != self.df_samp.shape[0]:
            num_missing = self.df_samp.shape[0] - num_tcrs_seen
            raise RuntimeError(
                f'Not all TCRs were seen over all trials (missing {num_missing} from '
                f'{self.df_samp.shape[0]} total). Increase the trial_count.'
            )

        min_sample_size = df_rand.select(
            pl.len().over(self.common_cols).min()
        )['len'].item(0)

        if min_sample_size == 1:
            raise RuntimeError(
                'Trials will be downsampled to 1 TCR enrichment, resulting in '
                'poor statistics. Rerun after increasing the trial_count.'
            )

        df_rand = df_rand.with_columns(
            idx=pl.int_range(pl.len()).over(self.common_cols)
        ).filter(
            # Downsample all TCRs to the same number.
            pl.col('idx') < min_sample_size
        ).group_by(
            pl.exclude('^*trial$', 'idx')
        ).agg(
            scores=pl.col('enrichment_trial').sort(),
            z_score=((pl.col('enrichment') - pl.col('enrichment_trial').mean())
                     / pl.col('enrichment_trial').std()).first(),
            ecdf=pl.int_range(0, pl.len() + 1) / pl.len(),
            search_sort=pl.col('enrichment_trial').sort().search_sorted(pl.col('enrichment')).first()
        ).with_columns(
            p_value=1 - pl.col('ecdf').list.get(pl.col('search_sort'))
        ).drop(
            ['ecdf', 'search_sort']
        ).sort(
            'index'
        )

        return df_rand

    def cluster(
        self,
        df: Optional[pl.DataFrame] = None,
        distance_vectorform: Optional[NDArray[np.int16]] = None,
        step: int = 5,
        init_breakpoint: float = 75,
        return_intermediates: bool = False,
        debug: bool = False,
        **kwargs: Dict[str, Any],
    ) -> pl.DataFrame | Tuple[pl.DataFrame, SegmentedLinearModel]:
        """
        Obtain the cluster of sequences around the most enriched sequence using
        segmented linear regression.

        Parameters
        ----------
        df : polars.DataFrame
            A DataFrame containing sequences and their enrichments.
        distance_vectorform : numpy.ndarray of numpy.int16
            The upper triangle of the TCRdist matrix of the sequences in the
            given DataFrame represented as a one-dimensional vector.
        step : int, default 5
            The width of the annuluses around the most enriched sequence.
        init_breakpoint: int, default 75
            The initial guess at which there is a breakpoint in the mean enrichment
            in an annulus vs. annulus radius.
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

            if not hasattr(self, 'sample_distance_vectorform'):
                raise RuntimeError(
                    'No enrichment values were computed for the original sample '
                    'repertoire, so no clusters can be created. Please run '
                    'compute_sample_enrichment() using the TCRTransport object '
                    'and then try running this function again.'
                )
            distance_vectorform = self.sample_distance_vectorform

        if 'enrichment' not in df.columns:
            raise RuntimeError(
                'No enrichment values were computed for the given input'
                'repertoire DataFrame, so a cluster cannot be create. Please '
                'compute the enrichment for the clones and then try running this '
                'function again.'
            )

        max_score_arg_max = df['enrichment'].arg_max()
        max_score = df[max_score_arg_max]['enrichment'].item(0)

        vectorform_idxs = square_indices_to_condensed_idx(
            np.zeros(df.shape[0], dtype=np.int32) + max_score_arg_max,
            np.arange(df.shape[0], dtype=np.int32),
            df.shape[0]
        )

        distance_vec = distance_vectorform[vectorform_idxs]
        distance_vec = np.insert(distance_vec, max_score_arg_max, 0)

        radii = np.arange(0, self.max_distance + step, step)
        # Determine which TCRs belong to which annulus.
        annulus_searchsort = np.searchsorted(radii, distance_vec, 'left')

        df = df.with_columns(
            enrichment_above_median=pl.col('enrichment') > pl.col('enrichment').quantile(0.5),
            dist_to_max_score=distance_vec,
            # Assign TCRs to annuluses.
            annulus_idx=annulus_searchsort
        ).filter(
            # Keep upper 50% enrichment TCRs only.
            (pl.col('enrichment_above_median'))
            # Remove TCRs beyond the max annulus radius.
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
        slm.fit(init_breakpoint, **kwargs)

        if slm.max0_params[0] < 1e-10:
            if debug:
                return tmp, slm
            raise RuntimeError(
                'Segmented linear model did not find a breakpoint.'
            )

        cluster_radius = slm.breakpoints[0]

        df = df.filter(
            (pl.col('dist_to_max_score') <= cluster_radius )
        ).drop(
            ('annulus_idx', 'annulus_radius', 'enrichment_above_median', 'dist_to_max_score')
        )

        if return_intermediates:
            return df, tmp, slm
        return df

    def create_clusters(
        self,
        max_cluster_count: int = 10,
        step: int = 5,
        init_breakpoint: float = 75,
        return_intermediates: bool = False,
        debug: bool = False,
        **kwargs
    ) -> pl.DataFrame | Tuple[pl.DataFrame, List[pl.DataFrame], List[SegmentedLinearModel]]:
        """
        Create clusters around each consecutive unclustered but most enriched sequence
        in the sample repertoire dataset.

        Parameters
        ----------
        max_cluster_count : int, default 10
            The number of clusters to form.
        step : int, default 5
            The width of the annuluses around the most enriched sequence. This
            is used to collect measurements and identify the breakpoint used
            to define a cluster.
        init_breakpoint: int, default 75
            The initial guess at which there is a breakpoint in the mean enrichment
            in an annulus vs. annulus radius.
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
            The DataFrame containing the sample repertoire sequences with an
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
        if 'enrichment' not in self.df_samp.columns:
            raise RuntimeError(
                'No enrichment values were computed for the original sample '
                'repertoire, so no clusters can be created. Please run '
                'compute_sample_enrichment() using the TCRTransport object '
                'and then try running this function again.'
            )

        tmp_dfs = []
        slms = []

        # Build the initial cluster from the complete sample repertoire.
        try:
            res = self.cluster(
                step=step, init_breakpoint=init_breakpoint, debug=debug,
                return_intermediates=return_intermediates, **kwargs,
            )
            if debug:
                if not isinstance(res, pl.DataFrame):
                    logging.info(
                        'Segmented linear model did not find a breakpoint. Returning '
                        'the DataFrame used to fit the model as well as the model object.'
                    )
                    return res
        except Exception as e:
            if 'did not find a breakpoint' in str(e):
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

        num_clusters = 1
        len_df_samp = len(self.df_samp)

        while num_clusters < max_cluster_count:
            # Obtain the subrepertoire which excludes all the previously
            # clustered TCRs.
            df_samp_sub = self.df_samp.join(
                df_cluster, on=self.common_cols, how='anti'
            )
            len_samp_sub = len(df_samp_sub)

            mass_samp_sub = get_mass_distribution(df_samp_sub, self.tcr_cols)

            # Get the subrepertoire's distance vectorform.
            samp_sub_idxs = df_samp_sub['index'].to_numpy()
            rows_sub, cols_sub = np.mask_indices(len_samp_sub, np.triu)
            rows, cols = samp_sub_idxs[rows_sub], samp_sub_idxs[cols_sub]
            vf_idxs = square_indices_to_condensed_idx(rows, cols, len_df_samp)
            distance_vectorform = self.sample_distance_vectorform[vf_idxs]

            # Get the subrepertoire's distance matrix.
            distance_matrix = self.distance_matrix[:, df_samp_sub['index']]

            effort_matrix = compute_effort_matrix(
                self.mass_ref, mass_samp_sub, distance_matrix, self.lambd
            )

            df_samp_sub = compute_enrichments(
                df_samp_sub, effort_matrix,
                distance_vectorform, self.max_distance, self.neighbor_radius
            )

            try:
                res = self.cluster(
                    df_samp_sub, distance_vectorform, step, init_breakpoint,
                    return_intermediates, debug, **kwargs
                )
                if debug:
                    if not isinstance(res, pl.DataFrame):
                        logging.info(
                            'Segmented linear model did not find a breakpoint. Returning '
                            'the DataFrame used to fit the model as well as the model object.'
                        )
                        return res
            except Exception as e:
                if 'did not find a breakpoint' in str(e):
                    logging.info(f'{e} Terminating finding clusters.')
                    break
                else:
                    raise e
            else:
                if return_intermediates:
                    df_samp_sub, tmp_df, slm = res
                    tmp_dfs.append(res[1])
                    slms.append(res[2])
                else:
                    df_samp_sub = res
                df_cluster = pl.concat((
                    df_cluster,
                    df_samp_sub.with_columns(
                        transport_cluster=pl.lit(num_clusters)
                    )
                ))

            num_clusters += 1

        self.df_samp =  self.df_samp.join(
            df_cluster, on=self.common_cols, how='left'
        ).drop(
            pl.col('^*right$')
        )

        if return_intermediates:
            return self.df_samp, tmp_dfs, slms
        return self.df_samp
