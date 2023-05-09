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

/// Calculates the pvalue of the z-score and adjusts for the alternative hypothesis
/// if necessary.
///
/// # Arguments
/// * `z_u` - The z-score of the U-Statistic
/// * `alternative` - The alternative hypothesis to test against
///
/// # Returns
/// * `pvalue` - The p-value of the test
fn p_value(z_u: f64, alternative: Alternative) -> f64 {
    let pvalue = Normal::new(0., 1.).unwrap().cdf(z_u);
    match alternative {
        Alternative::TwoSided => 2. * pvalue,
        _ => pvalue,
    }
}

/// Performs the Mann-Whitney U Test to measure the statistical difference between
/// two groups through their rank values.
///
/// # Arguments
/// * `x` - Array of values from the first group
/// * `y` - Array of values from the second group
/// * `alternative` - The alternative hypothesis to test against
/// * `use_continuity` - Whether to use continuity correction
pub fn mann_whitney_u(
    x: &Array1<f64>,
    y: &Array1<f64>,
    alternative: Alternative,
    use_continuity: bool,
) -> (f64, f64) {
    let (ranks_x, ranks_y) = merged_ranks(x, y);

    let nx = x.len() as f64;
    let ny = y.len() as f64;

    let u_t = alt_u_statistic(&ranks_x, &ranks_y, alternative);
    let z_u = z_score(u_t, nx, ny, use_continuity);
    let p_v = p_value(z_u, alternative);

    (u_t, p_v)
}

#[cfg(test)]
mod testing {
    use super::merged_ranks;
    use crate::mwu::mann_whitney_u;
    use ndarray::{array, Array};

    const EPSILON: f64 = 1e-10;

    fn test_close(a: f64, b: f64) {
        assert!((a - b).abs() < EPSILON);
    }

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
        let y = array![3., 5., 6.];
        assert_eq!(super::u_statistic(&x), 1.);
        assert_eq!(super::u_statistic(&y), 8.);
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
    fn test_alt_u_statistic_less() {
        let x = array![1., 2., 4.];
        let y = array![3., 5., 6.];
        assert_eq!(super::alt_u_statistic(&x, &y, super::Alternative::Less), 1.);
    }

    #[test]
    fn test_alt_u_statistic_greater() {
        let x = array![1., 2., 4.];
        let y = array![3., 5., 6.];
        assert_eq!(
            super::alt_u_statistic(&x, &y, super::Alternative::Greater),
            8.
        );
    }

    #[test]
    fn test_alt_u_statistic_two_sided() {
        let x = array![1., 2., 4.];
        let y = array![3., 5., 6.];
        assert_eq!(
            super::alt_u_statistic(&x, &y, super::Alternative::TwoSided),
            1.
        );
    }

    #[test]
    fn test_mann_whitney_u_twosided_continuity() {
        let x = Array::range(0.0, 10.0, 1.0);
        let y = Array::range(10.0, 20.0, 1.0);
        let (u, p) = mann_whitney_u(&x, &y, super::Alternative::TwoSided, true);
        assert_eq!(u, 0.);
        test_close(p, 0.0001826717911243235);
    }

    #[test]
    fn test_mann_whitney_u_twosided_no_continuity() {
        let x = Array::range(0.0, 10.0, 1.0);
        let y = Array::range(10.0, 20.0, 1.0);
        let (u, p) = mann_whitney_u(&x, &y, super::Alternative::TwoSided, false);
        assert_eq!(u, 0.);
        test_close(p, 0.00015705228423075119);
    }

    #[test]
    fn test_mann_whitney_u_less_continuity() {
        let x = Array::range(0.0, 10.0, 1.0);
        let y = Array::range(10.0, 20.0, 1.0);
        let (u, p) = mann_whitney_u(&x, &y, super::Alternative::Less, true);
        assert_eq!(u, 0.);
        test_close(p, 9.133589556216175e-5);
    }

    #[test]
    fn test_mann_whitney_u_less_no_continuity() {
        let x = Array::range(0.0, 10.0, 1.0);
        let y = Array::range(10.0, 20.0, 1.0);
        let (u, p) = mann_whitney_u(&x, &y, super::Alternative::Less, false);
        assert_eq!(u, 0.);
        test_close(p, 7.852614211537559e-05);
    }

    #[test]
    fn test_mann_whitney_u_greater_continuity() {
        let x = Array::range(0.0, 10.0, 1.0);
        let y = Array::range(10.0, 20.0, 1.0);
        let (u, p) = mann_whitney_u(&x, &y, super::Alternative::Greater, true);
        assert_eq!(u, 100.);
        test_close(p, 0.9999325785388173);
    }

    #[test]
    fn test_mann_whitney_u_greater_no_continuity() {
        let x = Array::range(0.0, 10.0, 1.0);
        let y = Array::range(10.0, 20.0, 1.0);
        let (u, p) = mann_whitney_u(&x, &y, super::Alternative::Greater, false);
        assert_eq!(u, 100.);
        test_close(p, 0.9999214738578847);
    }
}
