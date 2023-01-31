use ndarray::{concatenate, s, Array1, Axis};
use statrs::{
    distribution::{ContinuousCDF, Normal},
    statistics::{Data, OrderStatistics, RankTieBreaker},
};
use std::ops::Div;

/// Concatenate two arrays, rank them, and return the rankings for each of the groups.
fn merged_ranks(x: &Array1<f64>, y: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let midpoint = x.len();
    let joined = concatenate(Axis(0), &[x.view(), y.view()])
        .unwrap()
        .to_vec();
    let ranks = Array1::from_vec(Data::new(joined).ranks(RankTieBreaker::Average));
    (
        ranks.slice(s![..midpoint]).to_owned(),
        ranks.slice(s![midpoint..]).to_owned(),
    )
}

/// Calculates the U-Statistic given an array of ranks
fn u_statistic(array: &Array1<f64>) -> f64 {
    let s = array.sum();
    let n = array.len() as f64;
    s - ((n * (n + 1.)) / 2.)
}

/// Calculats the U-Distribution Mean
fn u_mean(nx: f64, ny: f64) -> f64 {
    (nx * ny) / 2.
}

/// Calculats the U-Distribution Standard Deviation
fn u_std(nx: f64, ny: f64) -> f64 {
    (nx * ny * (nx + ny + 1.)).div(12.).sqrt()
}

/// Performs the Mann-Whitney U Test otherwise known as the Rank-Sum Test to measure the
/// statistical difference between two values through their ranks.
pub fn mann_whitney_u(x: &Array1<f64>, y: &Array1<f64>) -> (f64, f64) {
    let (ranks_x, _ranks_y) = merged_ranks(x, y);

    let nx = x.len() as f64;
    let ny = y.len() as f64;

    let u_t = u_statistic(&ranks_x);
    let m_u = u_mean(nx, ny);
    let s_u = u_std(nx, ny);

    let z_u = (u_t - m_u) / s_u;

    (u_t, Normal::new(0., 1.).unwrap().cdf(z_u))
}
