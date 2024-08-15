import logging
from typing import *

import numpy as np
from numpy.typing import NDArray
import ot
import polars as pl

from tcr_transport.segmented_linear_model import SegmentedLinearModel
from tcr_transport.utils import *

logging.getLogger().setLevel(logging.INFO)

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
        A DataFrame containing the TCR recombinations. It is expected that this
        DataFrame was produced by the .It must have a
        'recomb_multiplicity' column.
    """
    if distribution_type == 'inverse_to_v_gene':
        pass
    elif distribution_type == 'uniform':
        return df.with_columns(
            tcr=pl.concat_str(cols)
        ).with_columns(
            mass=pl.col('recomb_multiplicity').sum().over('tcr') / pl.col('recomb_multiplicity').sum()
        )['mass'].to_numpy()
    else:
        raise RuntimeError(
            'Unsupported distribution_type. Valid values for distribution_type '
            'are \'inverse_to_v_gene\' and \'uniform\'.'
        )

def compute_effort_matrix(
    mass_reference: NDArray[np.float64],
    mass_sample: NDArray[np.float64],
    distance_matrix: NDArray[np.float64],
    lambd: float = 0.01,
) -> NDArray[np.float64]:
    ot_mat = ot.sinkhorn(mass_reference, mass_sample, distance_matrix, lambd)
    return distance_matrix * ot_mat

def compute_enrichments(
    df: pl.DataFrame,
    effort_matrix: NDArray[np.float64],
    distance_vectorform: NDArray[np.int16],
    max_distance: float = 200,
    neighbor_radius: int = 48,
    axis: int = 0,
):
    efforts = max_distance * df['recomb_multiplicity'].sum() * effort_matrix.sum(axis)

    len_df = df.shape[0]

    mask = np.zeros((len_df,) * 2, dtype=bool)
    np.fill_diagonal(mask, True)

    neighbor_idxs = np.where(distance_vectorform <= neighbor_radius)
    if neighbor_idxs:
        rows, cols = condensed_idx_to_square_indices(neighbor_idxs[0], len_df)
        mask[rows, cols] = mask[rows, cols] = True

    return df.with_columns(
        enrichment=efforts @ mask,
        num_neighbors=mask.sum(0)
    )

def split_datasets(
    df: pl.DataFrame,
    n: int,
    rng = None,
) -> Tuple[pl.DataFrame]:
    if rng is None:
        rng = np.random.default_rng()

    permutation = rng.permutation(df.shape[0])
    return  df[permutation[:n]], df[permutation[n:]]

class TCRTransport():
    def __init__(
        self,
        distribution_type: str = 'uniform',
        lambd: float = 0.01,
        max_distance: int = 200,
        neighbor_radius: int = 48,
        seed = None,
    ) -> None:
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
        **kwargs
    ) -> None:
        if beta_cols is None and alpha_cols is None:
            raise RuntimeError(
                'Both beta_cols and alpha_cols must not be None'
            )
        elif beta_cols is not None and alpha_cols is not None:
            seq_cols = self.tcr_cols[::2]
            v_cols = self.tcr_cols[1::2]
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
        )

        if reference:
            self.df_ref = df
        else:
            self.df_sample = df

        self.common_cols = self.tcr_cols + ['recomb_multiplicity']

    def compute_sample_enrichment(
        self,
    ) -> None:
        self.sample_distance_vectorform = compute_distance_vectorform(
            self.df_sample[self.tcr_cols].to_numpy()
        )
        self.mass_ref = get_mass_distribution(self.df_ref, self.tcr_cols)
        self.mass_samp = get_mass_distribution(self.df_sample, self.tcr_cols)
        distance_matrix = compute_distance_vectorform(
            self.df_ref[self.tcr_cols].to_numpy(),
            self.df_sample[self.tcr_cols].to_numpy(),
            dtype=np.float64
        ).reshape(self.df_ref.shape[0], self.df_sample.shape[0]) / self.max_distance

        self.effort_matrix = compute_effort_matrix(
            self.mass_ref, self.mass_samp, distance_matrix, self.lambd
        )

        self.df_sample = compute_enrichments(
            self.df_sample, self.effort_matrix, self.sample_distance_vectorform,
            self.max_distance, self.neighbor_radius
        )

    def do_randomization_test(
        self,
        trial_count: int = 100,
        seed = None
    ):
        if seed is not None:
            rng = self.rng
        else:
            rng = np.random.default_rng(seed)

        df_full = pl.concat((
            self.df_ref.select(self.common_cols), self.df_sample.select(self.common_cols)
        ))

        randomized_scores = []

        for trial in range(trial_count):
            df_ref_trial, df_samp_trial = split_datasets(df_full, self.df_ref.shape[0], rng)

            mass_ref_trial = get_mass_distribution(df_ref_trial, self.tcr_cols)
            mass_samp_trial = get_mass_distribution(df_samp_trial, self.tcr_cols)

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
                self.df_sample.join(
                    df_samp_trial,
                    on=self.common_cols,
                    suffix='_trial'
                )
            )

        df_rand = pl.concat(randomized_scores)

        num_tcrs_seen = df_rand.unique(self.common_cols).shape[0]
        if num_tcrs_seen != self.df_sample.shape[0]:
            num_missing = self.df_sample.shape[0] - num_tcrs_seen
            raise RuntimeError(
                f'Not all TCRs were seen over all trials (missing {num_missing} from '
                f'{self.df_sample.shape[0]} total). Increase the trial_count.'
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
            pl.exclude('enrichment_trial', 'num_neighbors_trial', 'idx')
        ).agg(
            scores=pl.col('enrichment_trial').sort(),
            z_score=((pl.col('enrichment') - pl.col('enrichment_trial').mean())
                     / pl.col('enrichment_trial').std()),
            ecdf=pl.int_range(0, pl.len() + 1) / pl.len(),
            search_sort=pl.col('enrichment_trial').sort().search_sorted(pl.col('enrichment')).first()
        ).with_columns(
            p_value=1 - pl.col('ecdf').list.get(pl.col('search_sort'))
        ).drop(
            ['ecdf', 'search_sort']
        )

        return df_rand

    def cluster(
        self,
        df: Optional[pl.DataFrame] = None,
        distance_vectorform: Optional[NDArray[np.int16]] = None,
        step: int = 5,
        init_breakpoint: float = 75,
        debug: bool = False,
        **kwargs
    ) -> pl.DataFrame:
        if df is None:
            df = self.df_sample
            distance_vectorform = self.sample_distance_vectorform

        max_score_tcr_index = df['enrichment'].arg_max()
        max_score = df[max_score_tcr_index]['enrichment'].item(0)

        enrichment_mask = df.select(
                pl.col('enrichment') > pl.col('enrichment').quantile(0.5)
        )['enrichment'].to_numpy()

        radii = np.arange(0, self.max_distance, step)

        neighborhood_mask = np.zeros(df.shape[0], dtype=bool)
        neighborhood_mask[max_score_tcr_index] = True

        vectorform_idxs = square_indices_to_condensed_idx(
            np.zeros(df.shape[0], dtype=np.int32) + max_score_tcr_index,
            np.arange(df.shape[0], dtype=np.int32),
            df.shape[0]
        )

        distance_vec = distance_vectorform[vectorform_idxs]
        distance_vec = np.insert(distance_vec, max_score_tcr_index, 0)

        mean_enrichments = np.zeros_like(radii, dtype=np.float64)
        annulus_enrichments = np.zeros_like(radii, dtype=np.float64)
        cluster_sizes = np.zeros_like(radii, dtype=np.int32)

        for idx, radius in enumerate(radii):
            neighbor_idxs = np.where(distance_vec <= radius)
            if neighbor_idxs:
                neighborhood_mask[neighbor_idxs] = True

            annulus_mask = (
                (distance_vec < radius + step)
                & (distance_vec >= radius)
                & enrichment_mask
            )
            full_mask = neighborhood_mask & enrichment_mask

            mean_neighborhood_enrichment = df.filter(
                full_mask
            )['enrichment'].mean()

            mean_annulus_enrichment = df.filter(
                annulus_mask
            )['enrichment'].mean()

            if mean_annulus_enrichment is None:
                mean_annulus_enrichment = np.nan

            mean_enrichments[idx] = mean_neighborhood_enrichment
            annulus_enrichments[idx] = mean_annulus_enrichment
            cluster_sizes[idx] = np.count_nonzero(full_mask)

        slm = SegmentedLinearModel()
        slm.fit(radii, annulus_enrichments, np.array([init_breakpoint]), **kwargs)

        if slm.max0_params[0] < 1e-10:
            if debug:
                return pl.DataFrame({
                    'radii': radii,
                    'mean_enrichments': mean_enrichments,
                    'annulus_enrichments': annulus_enrichments,
                    'cluster_sizes': cluster_sizes
                }), slm
            raise RuntimeError(
                'Segmented linear did not find a breakpoint.'
            )

        cluster_radius = slm.breakpoints[0]

        return df.filter(
            pl.lit(distance_vec <= cluster_radius)
        )

    def create_clusters(
        self,
        max_cluster_count: int = 10,
        step: int = 5,
        init_breakpoint: float = 75,
        debug: bool = False,
        **kwargs
    ) -> pl.DataFrame:
        # Build the initial cluster from the complete sample repertoire.
        df_cluster = self.cluster(
            step=step, init_breakpoint=init_breakpoint, debug=debug, **kwargs,
        )

        if not isinstance(df_cluster, pl.DataFrame):
            return df_cluster
        else:
            df_cluster = df_cluster.with_columns(
            transport_cluster=pl.lit(0)
        )
        num_clusters = 1

        while num_clusters <= max_cluster_count:
            df_samp_sub = self.df_sample.join(
                df_cluster, on=self.common_cols, how='anti'
            )
            mass_samp_sub = get_mass_distribution(df_samp_sub, self.tcr_cols)
            samp_sub_dist_vectorform = compute_distance_vectorform(
                df_samp_sub[self.tcr_cols].to_numpy()
            )

            distance_matrix = compute_distance_vectorform(
                self.df_ref[self.tcr_cols].to_numpy(),
                df_samp_sub[self.tcr_cols].to_numpy(),
                dtype=np.float64
            ).reshape(self.df_ref.shape[0], df_samp_sub[self.tcr_cols].shape[0]) / self.max_distance

            effort_matrix = compute_effort_matrix(
                self.mass_ref, mass_samp_sub, distance_matrix, self.lambd
            )

            df_samp_sub = compute_enrichments(
                df_samp_sub, effort_matrix, samp_sub_dist_vectorform,
                self.max_distance, self.neighbor_radius
            )

            try:
                df_samp_sub = self.cluster(
                    df_samp_sub, samp_sub_dist_vectorform, step, init_breakpoint,
                    debug=debug, **kwargs
                )

                if not isinstance(df_samp_sub, pl.DataFrame):
                    return df_samp_sub
            except Exception as e:
                if 'did not converge' in str(e):
                    logging.info(f'{e} Terminating finding clusters.')
                    break
                else:
                    raise e
            else:
                df_cluster = pl.concat((
                    df_cluster,
                    df_samp_sub.with_columns(
                    transport_cluster=pl.lit(num_clusters)
                )))
            num_clusters += 1

        return self.df_sample.join(
            df_cluster, on=self.common_cols, how='left'
        ).drop(
            pl.col('^*right$')
        )
