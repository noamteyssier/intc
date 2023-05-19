use crate::{
    mwu::{mann_whitney_u, Alternative},
    utils::select_values,
};
use ndarray::{s, Array1, Array2, Axis};
use ndarray_rand::{
    rand::{rngs::StdRng, SeedableRng},
    rand_distr::Uniform,
    RandomExt,
};

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
pub fn pseudo_rank_test_fast(
    n_pseudo: usize,
    s_pseudo: usize,
    ntc_pvalues: &Array1<f64>,
    ntc_logfcs: &Array1<f64>,
    alternative: Alternative,
    continuity: bool,
    seed: u64,
) -> (Array1<f64>, Array1<f64>) {
    let mut pseudo_pvalues = Array1::zeros(n_pseudo);
    let mut pseudo_logfc = Array1::zeros(n_pseudo);

    let num_ntc = ntc_pvalues.len();
    let mut rng = StdRng::seed_from_u64(seed);

    (0..n_pseudo)
        // generate array of random indices considered "in" the test group
        .map(|_| Array1::random_using(s_pseudo, Uniform::new(0, num_ntc), &mut rng))
        // subset the pvalues and logfc to the "in" and "out" groups
        .map(|mask| {
            let pvalues = ntc_pvalues.select(Axis(0), &mask.as_slice().unwrap());
            let logfcs = ntc_logfcs.select(Axis(0), &mask.as_slice().unwrap());
            (pvalues, logfcs)
        })
        .enumerate()
        // calculate the U, p-values, and aggregate logfc for each pseudo gene
        .for_each(|(idx, (ig_pvalues, ig_logfcs))| {
            let (_score, pvalue) =
                mann_whitney_u(&ig_pvalues, ntc_pvalues, alternative, continuity);
            let logfc = ig_logfcs.mean().unwrap_or(0.0);
            pseudo_pvalues[idx] = pvalue;
            pseudo_logfc[idx] = logfc;
        });

    (pseudo_pvalues, pseudo_logfc)
}

pub fn pseudo_rank_test_matrix(
    n_genes: usize,
    s_pseudo: usize,
    n_tests: usize,
    ntc_pvalues: &Array1<f64>,
    ntc_logfcs: &Array1<f64>,
    alternative: Alternative,
    continuity: bool,
    seed: u64,
) -> (Array2<f64>, Array2<f64>) {
    let mut pseudo_pvalues = Array2::zeros((n_tests, n_genes));
    let mut pseudo_logfc = Array2::zeros((n_tests, n_genes));

    (0..n_tests)
        .map(|idx| {
            let (pseudo_pvalues, pseudo_logfcs) = pseudo_rank_test_fast(
                n_genes,
                s_pseudo,
                ntc_pvalues,
                ntc_logfcs,
                alternative,
                continuity,
                seed + idx as u64,
            );
            (idx, pseudo_pvalues, pseudo_logfcs)
        })
        .for_each(|(idx, pvalues, logfcs)| {
            pseudo_logfc
                .slice_mut(s![idx, ..])
                .zip_mut_with(&logfcs, |x, &y| *x += y);
            pseudo_pvalues
                .slice_mut(s![idx, ..])
                .zip_mut_with(&pvalues, |x, &y| *x += y);
        });

    (pseudo_pvalues, pseudo_logfc)
}

#[cfg(test)]
mod testing {
    use super::rank_test;
    use crate::mwu::Alternative;
    use ndarray::array;

    #[test]
    fn test_rank_test() {
        let n_genes = 2;
        let ntc_index = 0;
        let encoding = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0];
        let test_values = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0];
        let ntc_values = array![0.7, 0.8, 0.9, 0.7, 0.8];
        let alternative = Alternative::TwoSided;
        let (u, p) = rank_test(
            n_genes,
            ntc_index,
            &encoding,
            &test_values,
            &ntc_values,
            alternative,
            false,
        );
        assert_eq!(u, vec![3.0, 4.5]);
        assert_eq!(p, vec![0.17971249488715443, 0.37109336955630695]);
    }

    #[test]
    fn test_pseudo_rank_test() {
        let n_pseudo = 2;
        let s_pseudo = 3;
        let ntc_pvalues = array![0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
        let ntc_logfcs = array![0.7, 0.7, 0.7, 0.7, 0.7, 0.7];
        let alternative = Alternative::TwoSided;
        let seed = 0;
        let (_u, p, _l) = super::pseudo_rank_test(
            n_pseudo,
            s_pseudo,
            &ntc_pvalues,
            &ntc_logfcs,
            alternative,
            false,
            seed,
        );
        assert_eq!(p, vec![1.0, 1.0]);
    }
}
