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
pub fn select_values(current_idx: usize, encodings: &[usize], values: &Array1<f64>) -> Array1<f64> {
    encodings
        .iter()
        .zip(values.iter())
        .filter(|(idx, _ranks)| **idx == current_idx)
        .map(|(_, value)| *value)
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
pub fn argsort<T>(array: &Array1<T>) -> Vec<usize>
where
    T: PartialOrd,
{
    let mut indices: Vec<usize> = (0..array.len()).collect();
    indices.sort_by(|&a, &b| array[a].partial_cmp(&array[b]).unwrap());
    indices
}

/// Performs an argsort on a 1D vector and returns an array of indices
pub fn argsort_vec<T>(vec: &Vec<T>) -> Vec<usize>
where
    T: PartialOrd,
{
    let mut indices: Vec<usize> = (0..vec.len()).collect();
    indices.sort_by(|&a, &b| vec[a].partial_cmp(&vec[b]).unwrap());
    indices
}

/// Calculates the diagonal product of fold changes and pvalues
pub fn diagonal_product(
    log2_fold_changes: &Array1<f64>,
    pvalues: &Array1<f64>,
) -> Array1<f64> {
    log2_fold_changes * pvalues.mapv(|x| x.exp2())
}

#[cfg(test)]
mod testing {
    use super::{argsort, argsort_vec};
    use hashbrown::HashMap;
    use ndarray::{array, Array1, Axis};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    #[test]
    fn test_argsort_forward() {
        let array = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let sorted = argsort(&array);
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        assert_eq!(
            array.select(Axis(0), &sorted),
            array![1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn test_argsort_reverse() {
        let array = array![5.0, 4.0, 3.0, 2.0, 1.0];
        let sorted = argsort(&array);
        assert_eq!(sorted, vec![4, 3, 2, 1, 0]);
        assert_eq!(
            array.select(Axis(0), &sorted),
            array![1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn test_reordering() {
        let pvalues = Array1::random(100, Uniform::new(0.0, 1.0));
        let order = argsort(&pvalues);
        let reorder = argsort_vec(&order);

        let sorted_pvalues = pvalues.select(Axis(0), &order);
        let resorted_pvalues = sorted_pvalues.select(Axis(0), &reorder);

        assert_ne!(pvalues, sorted_pvalues);
        assert_eq!(pvalues, resorted_pvalues);
    }

    #[test]
    fn test_select_values() {
        let encodings = vec![0, 0, 1, 1, 2, 2];
        let ranks = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let selected = super::select_values(1, &encodings, &ranks);
        assert_eq!(selected, array![0.3, 0.4]);
    }

    #[test]
    fn test_validate_token() {
        let mut map = HashMap::new();
        map.insert(0, "gene-0");
        map.insert(1, "gene-1");
        map.insert(2, "gene-2");
        map.insert(3, "gene-3");
        map.insert(4, "gene-4");
        let index = super::validate_token(&map, "gene-2").unwrap();
        assert_eq!(index, 2);
    }

    #[test]
    fn test_validate_token_duplicate() {
        let mut map = HashMap::new();
        map.insert(0, "gene-0");
        map.insert(1, "gene-1");
        map.insert(2, "gene-2");
        map.insert(3, "gene-3");
        map.insert(4, "gene-4");
        let index = super::validate_token(&map, "gene");
        assert!(index.is_err());
    }

    #[test]
    fn test_reconstruct_names() {
        let mut map = HashMap::new();
        map.insert(0, "gene-0");
        map.insert(1, "gene-1");
        map.insert(2, "gene-2");
        map.insert(3, "gene-3");
        map.insert(4, "gene-4");
        let names = super::reconstruct_names(&map, 2);
        assert_eq!(names, vec!["gene-0", "gene-1", "gene-3", "gene-4"]);
    }

    #[test]
    fn test_build_pseudo_names() {
        let names = super::build_pseudo_names(5);
        assert_eq!(
            names,
            vec![
                "pseudogene-0",
                "pseudogene-1",
                "pseudogene-2",
                "pseudogene-3",
                "pseudogene-4"
            ]
        );
    }
}
