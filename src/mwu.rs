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

#[cfg(test)]
mod testing {
    use crate::mwu::mann_whitney_u;

    use super::merged_ranks;
    use ndarray::{array, Array1};

    #[test]
    fn test_merged_ranks() {
        let x = array![1., 3., 5.];
        let y = array![2., 6., 4.];
        let (ranks_x, ranks_y) = merged_ranks(&x, &y);
        assert_eq!(ranks_x, array![1., 3., 5.]);
        assert_eq!(ranks_y, array![2., 6., 4.]);
    }

    #[test]
    fn test_u_statistic() {
        let x = array![1., 2., 4.];
        assert_eq!(super::u_statistic(&x), 1.);
    }

    #[test]
    fn test_u_mean() {
        assert_eq!(super::u_mean(3., 3.), 4.5);
    }

    #[test]
    fn test_u_std() {
        assert_eq!(super::u_std(3., 3.), 2.29128784747792);
    }

    #[test]
    fn test_mann_whitney_u() {
        let x = Array1::range(1., 6., 1.);
        let y = Array1::range(6., 11., 1.);
        println!("{x:?} {y:?}");
        let (_, pv) = mann_whitney_u(&x, &y);
        assert!(pv - 1.2185e-2 < 1e-6);
    }
}
