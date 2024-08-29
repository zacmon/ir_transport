from __future__ import annotations
from typing import *

import numpy as np
from numpy.typing import NDArray
import polars as pl
import tcrdist_rs as tdr

def load_data(
    data: str | Dict[str, Any] | Sequence | np.ndarray | pl.Series | pd.DataFrame | pl.DataFrame,
    seq_cols: str | Sequence[str],
    v_cols: str | Sequence[str] = None,
    collapse: Optional[str] = None,
    **kwargs
) -> pl.DataFrame:
    """
    Load the repertoire data into a polars.DataFrame and perform some preprocessing and filtering.

    Parameters
    ----------
    data : str or dict of {str : any} or Sequence or numpy.ndarray or polars.Series or pandas.DataFrame or polars.DataFrame
        A variable which points to sequence data. If a string, data must
        point to a valid file to be read. Otherwise, data could be an existing
        DataFrame or two-dimensional data in many forms.
    seq_cols : str or sequence of str
        String(s) which points to the alpha and/or beta CDR3 amino acid sequences.
    v_cols : str or sequence or str, optional
        String(s) which points to the alpha and/or beta V genes.
    collapse : str, optional
        If 'allele', the V genes have the allele information removed. If 'subfamily',
        the gene subfamily information is removed from the V annotation.
    **kwargs : dict of {str : any}
        Keyword arguments to polars.read_csv if data is a string or keyword arguments
        to polars.DataFrame if data is not a string.

    Returns
    -------
    df : polars.DataFrame
        A DataFrame containing the deduplicated sequence.
    """
    if collapse is not None and collapse not in {'allele', 'subfamily'}:
        raise ValueError(
            'collapse must be \'allele\' or \'subfamily\'.'
        )
    if isinstance(data, str):
        df = pl.read_csv(data, **kwargs)
    elif isinstance(data, pl.DataFrame):
        df = data
    else:
        df = pl.DataFrame(data, **kwargs)

    if isinstance(v_cols, str):
        v_cols = [v_cols]
    if isinstance(seq_cols, str):
        seq_cols = [seq_cols]

    # Ensure seq columns contain valid amino acids only.
    num_invalid_seqs = df.select(
        res=pl.sum_horizontal(
            ~pl.col(seq_cols).str.contains('^[ACDEFGHIKLMNPQRSTVWY~_\*]+$').all()
        )
    )['res'].item(0)

    if num_invalid_seqs > 0:
        raise RuntimeError(
            f'The column(s) pointed to by {seq_cols} contains invalid amino acid '
            'characters.'
        )

    if collapse is not None:
        if v_cols is None:
            raise RuntimeError(
                'v_cols must not be None if collapse it not None'
            )
        new_cols = {
            col: pl.col(col).str.replace_all(r'\*[0-9]+', '')
            for col in v_cols if col is not None
        }
        df = df.with_columns(
            **new_cols
        )
        if collapse == 'subfamily':
            new_cols = {
                col: pl.col(col).str.replace_all(r'\-[0-9]+', '')
                for col in v_cols if col is not None
            }
            df = df.with_columns(
                **new_cols
            )

    if v_cols is not None:
        grp_cols = list(v_cols) + list(seq_cols)
    else:
        grp_cols = seq_cols

    df = df.group_by(
        grp_cols
    ).agg(
        multiplicity=pl.len()
    )

    return df

def compute_distances(
    seqs: NDArray[str],
    seqs_comp: Optional[NDArray[str]] = None,
    species: str = 'human',
) -> NDArray[np.uint16]:
    """
    Compute the distances within a dataset of sequences or between two datasets.

    Parameters
    ----------
    seqs : numpy.ndarray of str
        A numpy array of strings.
    seqs_comp : numpy.ndarray of str, optional
        A numpy array of strings.
    species : str, default 'human'
        The species used for the V gene lookup table in TCRdist.

    Returns
    -------
    numpy.ndarray of numpy.uint16
        The computed TCRdists.
    """
    # TODO Add support for mice.
    # TODO Add support for CDR1, CDR2, CDR3 computations.
    if len(seqs.shape) == 1:
        dist_func = 'tcrdist_'
    elif seqs.shape[1] == 2:
        dist_func = 'tcrdist_gene_'
    else:
        dist_func = 'tcrdist_paired_gene_'

    if seqs_comp is not None:
        func_suffix = 'many_to_many'
        return np.fromiter(
            getattr(tdr, dist_func + func_suffix)(
                seqs, seqs_comp, parallel=True
            ), dtype=np.uint16
        ).reshape(
            seqs.shape[0], seqs_comp.shape[0]
        )
    else:
        func_suffix = 'matrix'
        return np.fromiter(
            getattr(tdr, dist_func + func_suffix)(
                seqs, parallel=True
            ), dtype=np.uint16
        )

def get_neighbor_map(
    seqs: NDArray[str],
    neighbor_radius: int,
    species: str = 'human',
) -> pl.DataFrame:
    """
    Return a DataFrame containing the indices of sequences which are neighbors
    and their distances.

    Parameters
    ----------
    seqs : numpy.ndarray of str
        A numpy array of strings.
    neighbor_radius : int
        The inclusive radius at which sequences are considered neighbors.
    species : str, default 'human'
        The species used for the V gene lookup table in TCRdist.

    Returns
    -------
    df_neighbors : polars.DataFrame
        A DataFrame containing the indices of sequences and their neighbors
        as well as the distance between the sequences.
    """
    # TODO Add support for mice.
    # TODO Add support for tcrdist, no gene, computations.
    # TODO Add support for CDR1, CDR2, CDR3 computations.
    if seqs.shape[1] == 2:
        dist_func = 'tcrdist_gene_'
    elif seqs.shape[1] == 4:
        dist_func = 'tcrdist_paired_gene_'
    else:
        raise RuntimeError(
            'seqs must either be the CDR3 and V allele/gene of a single chain '
            'or the CDR3 and V allele/gene for both chains.'
        )

    func_suffix = 'neighbor_matrix'

    neighbors = getattr(tdr, dist_func + func_suffix)(
        seqs, neighbor_radius, parallel=True
    )

    df_neighbors = pl.DataFrame(
        neighbors, schema=['index', 'n_index', 'dist'], orient='row',
        schema_overrides={
            'index': pl.UInt32, 'n_index': pl.UInt32, 'dist': pl.UInt16
        }
    )

    # Concatenate the index and n_index reversed to get all neighbors.
    df_neighbors = pl.concat((
        df_neighbors,
        df_neighbors.select(
            ['n_index', 'index', 'dist']
        ).rename({
            'n_index': 'index', 'index': 'n_index'
        })
    ))

    return df_neighbors

def square_indices_to_condensed_idx(
    rows: NDArray[np.integer],
    cols: NDArray[np.integer],
    n: int,
    dtype: str | type = np.int32
) -> NDArray[np.integer]:
    """
    Map indices from a symmetric square matrix to a condensed vectorform, i.e., the upper
    right triangle of the matrix.

    Formula from https://stackoverflow.com/a/36867493.

    Parameters
    ----------
    rows : numpy.ndarray of numpy.integer
        An array containing integers pointing to rows in the matrix.
    cols : numpy.ndarray of numpy.integer
        An array containing integers pointing to columns in the matrix.
    n : int
        The length of the square matrix.

    Returns
    -------
    numpy.ndarray of dtype
        The corresponding indices in the condensed vectorform.
    """
    # Remove diagonals.
    diagonals = rows == cols
    rows, cols = rows[~diagonals], cols[~diagonals]

    not_upper_tri = rows > cols
    rows[not_upper_tri], cols[not_upper_tri] = cols[not_upper_tri], rows[not_upper_tri]

    return (n * rows - rows * (rows + 1) // 2 + cols - 1 - rows).astype(dtype)

def condensed_idx_to_square_indices(
    idxs: NDArray[np.integer],
    n: int,
    dtype: str | type = np.int32
):
    """
    Map indices from a vectorform of the upper right triangular matrix to its
    symmetric squareform representation.

    Formulas from https://stackoverflow.com/a/36867493.

    Parameters
    ----------
    idxs : numpy.ndarray of numpy.integer
        Indices of the vectorform that will be mapped to their corresponding indices
        in the squareform matrix.
    n : int
        The length of the square matrix.
    dtype : str or type, default numpy.int64
        The dtype of the returned indices.

    Returns
    -------
    row_idxs : numpy.ndarray of numpy.integer
        The mapped row indices in the full square matrix.
    col_idxs : numpy.ndarray of numpy.integer
        The mapped column indices in the full square matrix.
    """
    row_idxs = np.ceil(0.5 * (2 * n - 1 - (4 * (n**2 - n - 2 * idxs) - 7)**0.5) - 1)
    row_idxs_p1 = row_idxs + 1
    col_idxs = n + idxs - (
        row_idxs_p1 * (n - 1 - row_idxs_p1)
        + (row_idxs_p1 * (row_idxs_p1 + 1)) // 2
    )
    return row_idxs.astype(dtype), col_idxs.astype(dtype)
