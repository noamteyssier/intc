use std::ops::Div;

use crate::utils::{
    argsort, argsort_vec, cumulative_sum, diagonal_product, diagonal_product_matrix,
};
use ndarray::{s, Array1, Array2, ArrayView1, Axis};

const EPSILON: f64 = 1e-10;

#[derive(Clone, Copy, Debug)]
pub enum Direction {
    Less,
    Greater,
}

#[derive(Debug)]
/// Result struct for False Discovery Rate and p-value threshold
pub struct FdrResult {
    fdr: Array1<f64>,
    threshold: f64,
}
impl FdrResult {
    /// Create a new FdrResult
    ///
    /// # Arguments
    /// * `fdr` - Array of FDR values
    /// * `threshold` - p-value threshold
    ///
    /// # Returns
    /// FdrResult
    pub fn new(fdr: Array1<f64>, threshold: f64) -> Self {
        Self { fdr, threshold }
    }

    /// Get the FDR values
    pub fn fdr(&self) -> &Array1<f64> {
        &self.fdr
    }

    /// Get the p-value threshold
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

#[derive(Debug)]
/// False Discovery Rate
pub struct Fdr<'a> {
    pvalues: &'a Array1<f64>,
    product: Option<Array1<f64>>,
    matrix_pvalues: &'a Array2<f64>,
    matrix_product: Option<Array2<f64>>,
    alpha: f64,
    use_product: Option<Direction>,
    n_draws: usize,
}

impl<'a> Fdr<'a> {
    /// Create a new FDR struct
    pub fn new(
        pvalues: &'a Array1<f64>,
        logfc: &'a Array1<f64>,
        matrix_pvalues: &'a Array2<f64>,
        matrix_logfc: &'a Array2<f64>,
        alpha: f64,
        use_product: Option<Direction>,
    ) -> Self {
        let product = if use_product.is_some() {
            Some(diagonal_product(logfc, pvalues))
        } else {
            None
        };
        let matrix_product = if use_product.is_some() {
            Some(diagonal_product_matrix(&matrix_logfc, &matrix_pvalues))
        } else {
            None
        };
        Self {
            pvalues,
            matrix_pvalues,
            alpha,
            use_product,
            product,
            matrix_product,
            n_draws: matrix_pvalues.nrows(),
        }
    }

    /// Creates a binary vector sorted so the first `size_real` elements are 0 and the rest 1
    fn build_typevec(size_real: usize, size_fake: usize) -> Array1<f64> {
        let n = size_real + size_fake;
        (0..n)
            .map(|i| if i < size_real { 0.0 } else { 1.0 })
            .collect::<Array1<f64>>()
    }

    /// Concatenates two vectors
    fn concatenate_values(real: &Array1<f64>, fake: &ArrayView1<f64>) -> Array1<f64> {
        real.iter()
            .chain(fake.iter())
            .cloned()
            .collect::<Array1<f64>>()
    }

    /// calculates the empirical false discovery rate of a given set of bools
    /// assumes the positive values are the fake ones.
    fn calculate_fdr(n: usize, sorted_boolvec: &Array1<f64>) -> Array1<f64> {
        let ranks = Array1::range(1.0, n as f64 + 1.0, 1.0);
        let cumulative_sum = cumulative_sum(sorted_boolvec);
        let fdr = cumulative_sum.div(&ranks);
        fdr
    }

    /// Calculates the threshold value for a given false discovery rate
    fn calculate_threshold(
        values: &Array1<f64>,
        fdr: &Array1<f64>,
        alpha: f64,
        use_product: Option<Direction>,
    ) -> f64 {
        if let Some(idxmin) = fdr.iter().position(|&x| x > alpha) {
            if idxmin == 0 {
                match use_product {
                    Some(Direction::Greater) => values[0] + EPSILON,
                    Some(Direction::Less) => values[0] - EPSILON,
                    None => (values[0] - EPSILON).max(0.0),
                }
            } else {
                values[idxmin]
            }
        } else {
            values[values.len() - 1]
        }
    }

    /// Calculates the empirical fdr for a given set of p-values and logfc
    fn empirical_fdr(&self, real: &Array1<f64>, fake: &ArrayView1<f64>) -> (Array1<f64>, f64) {
        let boolvec = Self::build_typevec(real.len(), fake.len());
        let allvec = Self::concatenate_values(real, fake);

        let order = argsort(&allvec, true);
        let reorder = argsort_vec(&order);

        let sorted_boolvec = boolvec.select(Axis(0), &order);
        let sorted_allvec = allvec.select(Axis(0), &order);

        let fdr = Self::calculate_fdr(allvec.len(), &sorted_boolvec);
        let threshold =
            Self::calculate_threshold(&sorted_allvec, &fdr, self.alpha, self.use_product);
        let unsorted_fdr = fdr.select(Axis(0), &reorder);
        let unsorted_fdr = unsorted_fdr.slice_move(s![..=real.len()]);

        (unsorted_fdr, threshold)
    }

    /// Fit the FDR
    pub fn fit(&self) -> FdrResult {
        let mut fdr_matrix = Array2::zeros(self.matrix_pvalues.dim());
        let mut threshold_arr = Array1::zeros(self.matrix_pvalues.dim().0);

        if let Some(_) = self.use_product {
            (0..self.n_draws).for_each(|i| {
                let (fdr, threshold) = self.empirical_fdr(
                    &self.product.as_ref().unwrap(),
                    &self.matrix_product.as_ref().unwrap().row(i),
                );
                fdr_matrix.row_mut(i).assign(&fdr);
                threshold_arr[i] = threshold;
            })
        } else {
            (0..self.n_draws).for_each(|i| {
                let (fdr, threshold) =
                    self.empirical_fdr(&self.pvalues, &self.matrix_pvalues.row(i));
                fdr_matrix.row_mut(i).assign(&fdr);
                threshold_arr[i] = threshold;
            });
        }

        let unsorted_fdr = fdr_matrix
            .mean_axis(Axis(0))
            .expect("Could not calculate INTC mean fdr");
        let threshold = threshold_arr
            .mean()
            .expect("Could not calculate INTC threshold");

        FdrResult::new(unsorted_fdr, threshold)
    }
}

#[cfg(test)]
mod testing {
    use crate::fdr::{Direction, EPSILON};

    use super::Fdr;
    use ndarray::{array, Array1};

    #[test]
    fn test_fdr() {
        let pvalues = array![0.1, 0.2, 0.3];
        let logfc = array![0.1, 0.2, 0.3];
        let ntc_indices = vec![1];
        let alpha = 0.1;
        let fdr = Fdr::new(&pvalues, &logfc, &ntc_indices, alpha, None).fit();
        assert_eq!(fdr.fdr(), array![0.0, 0.5, 1. / 3.]);
    }

    #[test]
    fn test_fdr_unsorted() {
        let pvalues = array![0.2, 0.1, 0.3];
        let logfc = array![0.1, 0.2, 0.3];
        let ntc_indices = vec![1];
        let alpha = 0.1;
        let fdr = Fdr::new(&pvalues, &logfc, &ntc_indices, alpha, None).fit();
        assert_eq!(fdr.fdr(), array![0.5, 1.0, 1. / 3.]);
    }

    #[test]
    fn test_fdr_unsorted_larger() {
        let pvalues = array![0.5, 0.1, 0.3, 0.4, 0.2, 0.6];
        let logfc = array![0.5, 0.1, 0.3, 0.4, 0.2, 0.6];
        let ntc_indices = vec![3, 5];
        let alpha = 0.1;
        let fdr = Fdr::new(&pvalues, &logfc, &ntc_indices, alpha, None).fit();
        assert_eq!(fdr.fdr(), array![0.2, 0.0, 0.0, 0.25, 0.0, 1. / 3.]);
    }

    #[test]
    fn test_ntc_mask() {
        let ntc_indices = vec![1];
        let mask = Fdr::ntc_mask(&ntc_indices, 3);
        assert_eq!(mask, array![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_fdr_threshold() {
        let m = 10;
        let pvalues = Array1::linspace(0.0, 1.0, m);
        let logfc = Array1::linspace(0.0, 1.0, m);
        let ntc_indices = vec![4, 6, 9];
        let alpha = 0.1;
        let fdr = Fdr::new(&pvalues, &logfc, &ntc_indices, alpha, None).fit();
        assert_eq!(fdr.threshold(), 1. / 3.);
    }

    #[test]
    fn test_fdr_threshold_direction_lt_saturated() {
        let m = 10;
        let pvalues = Array1::linspace(0.1, 1.0, m);
        let logfc = Array1::linspace(-1.0, -0.01, m);
        let ntc_indices = vec![0, 3, 5];
        let alpha = 0.1;
        let fdr = Fdr::new(&pvalues, &logfc, &ntc_indices, alpha, Some(Direction::Less)).fit();
        assert_eq!(fdr.threshold(), -(0.1f64).log10() * -1.0 - EPSILON);
    }

    #[test]
    fn test_fdr_threshold_direction_gt_saturated() {
        let m = 10;
        let pvalues = Array1::linspace(0.1, 1.0, m);
        let logfc = Array1::linspace(0.1, 1.0, m)
            .iter()
            .rev()
            .cloned()
            .collect::<Array1<f64>>();
        let ntc_indices = vec![0, 3, 5];
        let alpha = 0.1;
        let fdr = Fdr::new(
            &pvalues,
            &logfc,
            &ntc_indices,
            alpha,
            Some(Direction::Greater),
        )
        .fit();
        assert_eq!(fdr.threshold(), -(0.1f64).log10() * 1.0 + EPSILON);
    }

    #[test]
    fn test_fdr_threshold_saturated() {
        let m = 10;
        let pvalues = Array1::linspace(0.1, 1.0, m);
        let logfc = Array1::linspace(-1.0, -0.01, m);
        let ntc_indices = vec![0, 3, 5];
        let alpha = 0.1;
        let fdr = Fdr::new(&pvalues, &logfc, &ntc_indices, alpha, None).fit();
        assert_eq!(fdr.threshold(), 0.1f64 - EPSILON);
    }

    #[test]
    fn test_fdr_threshold_saturated_epsilon() {
        let m = 10;
        let pvalues = Array1::linspace(EPSILON, 1.0, m);
        let logfc = Array1::linspace(-1.0, -0.01, m);
        let ntc_indices = vec![0, 3, 5];
        let alpha = 0.1;
        let fdr = Fdr::new(&pvalues, &logfc, &ntc_indices, alpha, None).fit();
        assert_eq!(fdr.threshold(), 0.0);
    }

    #[test]
    fn test_fdr_threshold_saturated_nonzero() {
        let m = 10;
        let pvalues = Array1::linspace(f64::EPSILON, 1.0, m);
        let logfc = Array1::linspace(-1.0, -0.01, m);
        let ntc_indices = vec![0, 3, 5];
        let alpha = 0.1;
        let fdr = Fdr::new(&pvalues, &logfc, &ntc_indices, alpha, None).fit();
        assert_eq!(fdr.threshold(), 0.0);
    }
}
