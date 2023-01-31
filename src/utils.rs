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

/// Builds a vector of gene names from the provided map skipping the non-targeting control index
pub fn reconstruct_names(map: &HashMap<usize, &str>, ntc_index: usize) -> Vec<String> {
    (0..map.len())
        .filter(|x| *x != ntc_index)
        .map(|x| map.get(&x).unwrap().to_string())
        .collect()
}

/// Builds a vector of pseudo gene names
pub fn build_pseudo_names(n_pseudo: usize) -> Vec<String> {
    (0..n_pseudo).map(|x| format!("pseudogene-{}", x)).collect()
}

/// Performs an argsort on a 1D ndarray and returns an array of indices
pub fn argsort(array: &Array1<f64>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..array.len()).collect();
    indices.sort_by(|&a, &b| array[a].partial_cmp(&array[b]).unwrap());
    indices
}

#[cfg(test)]
mod testing {
    use super::argsort;
    use ndarray::array;

    #[test]
    fn test_argsort_forward() {
        let array = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let sorted = argsort(&array);
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_argsort_reverse() {
        let array = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let sorted = argsort(&array);
        assert_eq!(sorted, vec![4, 3, 2, 1, 0]);
    }
}
