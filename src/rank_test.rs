use crate::{
    mwu::{mann_whitney_u, Alternative},
    utils::select_values,
};
use ndarray::{Array1, Axis};
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
pub fn pseudo_rank_test(
    n_pseudo: usize,
    s_pseudo: usize,
    ntc_pvalues: &Array1<f64>,
    ntc_logfcs: &Array1<f64>,
    alternative: Alternative,
    continuity: bool,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut pseudo_scores = Vec::with_capacity(n_pseudo);
    let mut pseudo_pvalues = Vec::with_capacity(n_pseudo);
    let mut pseudo_logfc = Vec::with_capacity(n_pseudo);
    let num_ntc = ntc_pvalues.len();
    let mut rng = StdRng::seed_from_u64(seed);

    (0..n_pseudo)
        // generate array of random indices considered "in" the test group
        .map(|_| Array1::random_using(s_pseudo, Uniform::new(0, num_ntc), &mut rng))
        // generate the complement list of indices considered "out" of the test group
        .map(|mask| {
            let slice_mask = mask.as_slice().unwrap().to_owned();
            let out_mask = (0..num_ntc)
                .filter(|x| !slice_mask.contains(x))
                .collect::<Vec<usize>>();
            (slice_mask, out_mask)
        })
        // subset the pvalues and logfc to the "in" and "out" groups
        .map(|(in_mask, out_mask)| {
            let in_group_pvalues = ntc_pvalues.select(Axis(0), &in_mask);
            let in_group_logfcs = ntc_logfcs.select(Axis(0), &in_mask);
            let out_group_pvalues = ntc_pvalues.select(Axis(0), &out_mask);
            (in_group_pvalues, in_group_logfcs, out_group_pvalues)
        })
        // calculate the U, p-values, and aggregate logfc for each pseudo gene
        .for_each(|(ig_pvalues, ig_logfcs, og_pvalues)| {
            let (score, pvalue) = mann_whitney_u(&ig_pvalues, &og_pvalues, alternative, continuity);
            let logfc = ig_logfcs.mean().unwrap_or(0.0);

            pseudo_scores.push(score);
            pseudo_pvalues.push(pvalue);
            pseudo_logfc.push(logfc);
        });

    (pseudo_scores, pseudo_pvalues, pseudo_logfc)
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
