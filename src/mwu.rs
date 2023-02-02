use ndarray::{concatenate, s, Array1, Axis};
use statrs::{
    distribution::{ContinuousCDF, Normal},
    statistics::{Data, OrderStatistics, RankTieBreaker},
};
use std::ops::Div;

/// Defines the alternative hypothesis 
#[derive(Clone, Copy, Default, Debug)]
pub enum Alternative {

    /// The alternative hypothesis is that the first array is greater than the second array
    Greater,

    /// The alternative hypothesis is that the first array is less than the second array
    Less,

    /// The alternative hypothesis is that the first array is not equal to the second array
    #[default]
    TwoSided,
}

/// Concatenate two arrays, rank them, and return the rankings for each of the groups.
///
/// # Arguments
/// * `x` - The first array
/// * `y` - The second array
///
/// # Returns
/// * `ranks` - A tuple containing the rankings for each of the groups
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
///
/// # Arguments
/// * `array` - The array of ranks
///
/// # Returns
/// * `u` - The U-Statistic
fn u_statistic(array: &Array1<f64>) -> f64 {
    let s = array.sum();
    let n = array.len() as f64;
    s - ((n * (n + 1.)) / 2.)
}

/// Calculates the Mann-Whitney U-Statistic
///
/// # Arguments
/// * `x` - The first group of ranks
/// * `y` - The second group of ranks
/// * `alternative` - The alternative hypothesis
///
/// # Returns
/// * `u` - The U-Statistic
fn alt_u_statistic(ranks_x: &Array1<f64>, ranks_y: &Array1<f64>, alternative: Alternative) -> f64 {
    match alternative {
        Alternative::Less => u_statistic(ranks_x),
        Alternative::Greater => u_statistic(ranks_y),
        Alternative::TwoSided => {
            let u_x = u_statistic(ranks_x);
            let u_y = u_statistic(ranks_y);
            u_x.min(u_y)
        }
    }
}

/// Calculates the U-Distribution Mean
///
/// # Arguments
/// * `nx` - The number of elements in the first group
/// * `ny` - The number of elements in the second group
///
/// # Returns
/// * `m_u` - The mean of the U-Distribution
fn u_mean(nx: f64, ny: f64) -> f64 {
    (nx * ny) / 2.
}

/// Calculates the U-Distribution Standard Deviation
///
/// # Arguments
/// * `nx` - The number of elements in the first group
/// * `ny` - The number of elements in the second group
///
/// # Returns
/// * `s_u` - The standard deviation of the U-Distribution
fn u_std(nx: f64, ny: f64) -> f64 {
    (nx * ny * (nx + ny + 1.)).div(12.).sqrt()
}

/// Calculates the Z-Score of the U-Statistic
//
// Continuity correction.
// Because SF is always used to calculate the p-value, we can always
// _subtract_ 0.5 for the continuity correction. This always increases the
// p-value to account for the rest of the probability mass _at_ q = U.
//
// # Arguments
// * `u` - The U-Statistic
// * `nx` - The number of elements in the first group
// * `ny` - The number of elements in the second group
// * `use_continuity` - Whether to use continuity correction
//
// # Returns
// * `z_u` - The z-score of the U-Statistic
fn z_score(u: f64, nx: f64, ny: f64, use_continuity: bool) -> f64 {
    let m_u = u_mean(nx, ny);
    let s_u = u_std(nx, ny);

    if use_continuity {
        (u - m_u + 0.5) / s_u
    } else {
        (u - m_u) / s_u
    }

}

/// Performs the Mann-Whitney U Test otherwise known as the Rank-Sum Test to measure the
/// statistical difference between two values through their ranks.
pub fn mann_whitney_u(x: &Array1<f64>, y: &Array1<f64>, use_continuity: bool) -> (f64, f64) {
    let (ranks_x, _ranks_y) = merged_ranks(x, y);

    let nx = x.len() as f64;
    let ny = y.len() as f64;

    let u_t = u_statistic(&ranks_x);
    let z_u = z_score(u_t, nx, ny, use_continuity);

    (u_t, Normal::new(0., 1.).unwrap().cdf(z_u))
}

#[cfg(test)]
mod testing {
    use crate::mwu::mann_whitney_u;

    use super::merged_ranks;
    use ndarray::{array, Array1};

    #[test]
    fn test_merged_ranks() {
        let x = array![10., 20., 50.];
        let y = array![30., 40., 60.];
        let (ranks_x, ranks_y) = merged_ranks(&x, &y);
        assert_eq!(ranks_x, array![1., 2., 5.]);
        assert_eq!(ranks_y, array![3., 4., 6.]);
    }

    #[test]
    fn test_merged_ranks_tie() {
        let x = array![10., 20., 50., 50.];
        let y = array![30., 40., 60., 60.];
        let (ranks_x, ranks_y) = merged_ranks(&x, &y);
        assert_eq!(ranks_x, array![1., 2., 5.5, 5.5]);
        assert_eq!(ranks_y, array![3., 4., 7.5, 7.5]);
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
    fn test_z_score_continuity() {
        assert_eq!(super::z_score(1., 3., 3., true), -1.3093073414159544);
    }

    #[test]
    fn test_z_score_no_continuity() {
        assert_eq!(super::z_score(1., 3., 3., false), -1.5275252316519468);
    }

    #[test]
    fn test_mann_whitney_u() {
        let x = Array1::range(1., 100., 1.);
        let y = Array1::range(100., 200., 1.);
        let (_, pv) = mann_whitney_u(&x, &y, false);
        assert!(pv - 1.87e-34 < 1e-30);
    }
}
