use anyhow::{bail, Result};
use hashbrown::HashMap;
use ndarray::Array1;

/// Validates the provided token is found one and only once in the gene set
pub fn validate_token(encode_map: &HashMap<usize, &str>, token: &str) -> Result<usize> {
    let ntc_index = encode_map
        .iter()
        .filter(|(_idx, gene)| gene.contains(token))
        .map(|(idx, _gene)| *idx)
        .collect::<Vec<usize>>();

    if ntc_index.len() != 1 {
        bail!("Multiple potential genes found with provided non-targeting control token")
    }
    Ok(ntc_index[0])
}

/// Select the ranks for a provided embedding. Applies a filter which selects all ranks
/// for the current gene index
pub fn select_ranks(current_idx: usize, encodings: &[usize], ranks: &Array1<f64>) -> Array1<f64> {
    encodings
        .iter()
        .zip(ranks.iter())
        .filter(|(idx, _ranks)| **idx == current_idx)
        .map(|(_, ranks)| *ranks)
        .collect()
}
