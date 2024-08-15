from __future__ import annotations
from typing import *

import numpy as np
from numpy.typing import NDArray
import polars as pl
import tcrdist_rs as tdr

def load_data(
    data: str | Dict[str, Any] | Sequence | np.ndarray | pl.Series | pd.DataFrame | pl.DataFrame,
    seq_cols: str | Sequence[str],
    v_cols: str | Sequence[str],
    collapse: Optional[str] = None,
    **kwargs
) -> pl.DataFrame:
    """
    Load the repertoire data into a polars.DataFrame and perform some preprocessing and filtering.

    Parameters
    ----------
    data : str or dict of {str : any} or Sequence or numpy.ndarray or polars.Series or pandas.DataFrame or polars.DataFrame
        A variable which points to TCR repertoire data. If a string, data must
        point to a valid file to be read. Otherwise, data could be an existing
        DataFrame or two-dimensional data in many forms.
    seq_cols : str or sequence of str
        String(s) which points to the alpha and/or beta CDR3 amino acid columns in the DataFrame.
    v_cols : str or sequence or str
        String(s) which points to the alpha and/or beta V genes.
    collapse : str, optional
        If 'gene', the V genes have the allele information removed. If 'subfamily',
        the gene information is removed from the V annotation.
    **kwargs : dict of {str : any}
        Keyword arguments to polars.read_csv if data is a string or keyword arguments
        to polars.DataFrame if data is not a string.

    Returns
    -------
    df : polars.DataFrame
        A DataFrame containing the TCR repertoire annotations deduplicated at the
        level of recombinations.
    """
    if collapse is not None and collapse not in {'gene', 'subfamily'}:
        raise ValueError(
            'collapse must be \'gene\' or \'subfamily\'.'
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
                'v_cols must not be None is collapse it not None'
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

    df = df.group_by(
        list(v_cols) + list(seq_cols)
    ).agg(
        recomb_multiplicity=pl.len()
    )

    return df

def compute_distance_vectorform(
    seqs: NDArray[str],
    seqs_comp: Optional[NDArray[str]] = None,
    dtype: str | type = np.int16,
) -> NDArray[np.uint16]:
    if seqs.shape[1] == 2:
        dist_func = f'tcrdist_gene_'
    else:
        dist_func = 'tcrdist_paired_gene_'

    if seqs_comp is not None:
        func_suffix = 'many_to_many'
        return np.fromiter(
            getattr(tdr, dist_func + func_suffix)(
                seqs, seqs_comp, parallel=True
            ), dtype=dtype
        )
    else:
        func_suffix = 'matrix'
        return np.fromiter(
            getattr(tdr, dist_func + func_suffix)(
                seqs, parallel=True
            ), dtype=dtype
        )

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
    col_idxs = n + idxs - (row_idxs_p1 * (n - 1 - row_idxs_p1) + (row_idxs_p1 * (row_idxs_p1 + 1)) // 2)
    return row_idxs.astype(dtype), col_idxs.astype(dtype)
