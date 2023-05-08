use crate::{mwu::{mann_whitney_u, Alternative}, utils::select_ranks};
use ndarray::{Array1, Axis};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

/// Performs a rank test for each gene in the dataset.
/// Returns a tuple of vectors containing the U and p-values for each gene.
pub fn rank_test(
    n_genes: usize,
    ntc_index: usize,
    encoding: &[usize],
    test_values: &Array1<f64>,
    ntc_values: &Array1<f64>,
    alternative: Alternative,
    continuity: bool,
) -> (Vec<f64>, Vec<f64>) {
    (0..=n_genes)
        .filter(|x| *x != ntc_index)
        .map(|x| select_values(x, encoding, test_values))
        .map(|x| mann_whitney_u(&x, ntc_values, alternative, continuity))
        .unzip()
}

/// Performs a rank test for pseudo genes created from the non-targeting controls
/// Returns a tuple of vectors containing the U and p-values for each pseudogene.
pub fn pseudo_rank_test(
    n_pseudo: usize,
    s_pseudo: usize,
    ntc_values: &Array1<f64>,
    alternative: Alternative,
    continuity: bool,
) -> (Vec<f64>, Vec<f64>) {
    (0..n_pseudo)
        .map(|_| Array1::random(s_pseudo, Uniform::new(0, ntc_values.len())))
        .map(|mask| {
            let slice_mask = mask.as_slice().unwrap();
            let out_mask = (0..ntc_values.len())
                .filter(|x| !slice_mask.contains(x))
                .collect::<Vec<usize>>();
            let in_group = ntc_values.select(Axis(0), slice_mask);
            let out_group = ntc_values.select(Axis(0), &out_mask);
            (in_group, out_group)
        })
        .map(|(in_group, out_group)| mann_whitney_u(&in_group, &out_group, alternative, continuity))
        .unzip()
}

#[cfg(test)]
mod testing {
    use super::rank_test;
    use ndarray::array;
    use crate::mwu::Alternative;

    #[test]
    fn test_rank_test() {
        let n_genes = 2;
        let ntc_index = 0;
        let encoding = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0];
        let pvalues = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0];
        let ntc_values = array![0.7, 0.8, 0.9, 0.7, 0.8];
        let alternative = Alternative::TwoSided;
        let (u, p) = rank_test(n_genes, ntc_index, &encoding, &pvalues, &ntc_values, alternative, false);
        assert_eq!(u, vec![3.0, 4.5]);
        assert_eq!(p, vec![0.17971249488715443, 0.37109336955630695]);
    }

    #[test]
    fn test_pseudo_rank_test() {
        let n_pseudo = 2;
        let s_pseudo = 3;
        let ntc_values = array![0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
        let alternative = Alternative::TwoSided;
        let (_u, p) = super::pseudo_rank_test(n_pseudo, s_pseudo, &ntc_values, alternative, false);
        assert_eq!(p, vec![1.0, 1.0]);
    }
}
