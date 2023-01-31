use crate::{mwu::mann_whitney_u, utils::select_ranks};
use ndarray::{Array1, Axis};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

pub fn rank_test(
    n_genes: usize,
    ntc_index: usize,
    encoding: &[usize],
    pvalues: &Array1<f64>,
    ntc_values: &Array1<f64>,
) -> (Vec<f64>, Vec<f64>) {
    (0..=n_genes)
        .filter(|x| *x != ntc_index)
        .map(|x| select_ranks(x, encoding, pvalues))
        .map(|x| mann_whitney_u(&x, ntc_values))
        .unzip()
}

pub fn pseudo_rank_test(
    n_pseudo: usize,
    s_pseudo: usize,
    ntc_values: &Array1<f64>,
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
        .map(|(in_group, out_group)| mann_whitney_u(&in_group, &out_group))
        .unzip()
}
