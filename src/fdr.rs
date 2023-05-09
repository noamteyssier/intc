use crate::utils::{argsort, argsort_vec, diagonal_product};
use ndarray::{Array1, Axis};

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
    logfc: &'a Array1<f64>,
    product: Array1<f64>,
    ntc_indices: &'a [usize],
    alpha: f64,
    use_product: Option<Direction>,
}

impl<'a> Fdr<'a> {
    /// Create a new FDR struct
    pub fn new(
        pvalues: &'a Array1<f64>,
        logfc: &'a Array1<f64>,
        ntc_indices: &'a [usize],
        alpha: f64,
        use_product: Option<Direction>,
    ) -> Self {
        Self {
            pvalues,
            logfc,
            ntc_indices,
            alpha,
            use_product,
            product: diagonal_product(logfc, pvalues),
        }
    }

    /// Fit the FDR
    pub fn fit(&self) -> FdrResult {
        let values = match self.use_product {
            Some(Direction::Less) => &self.product,
            Some(Direction::Greater) => &self.product,
            None => &self.logfc,
        };
        let order = match self.use_product {
            Some(Direction::Less) => argsort(&self.product, true),
            Some(Direction::Greater) => argsort(&self.product, false),
            None => argsort(&self.pvalues, true),
        };
        let reorder = argsort_vec(&order);
        let is_ntc = Self::ntc_mask(self.ntc_indices, self.pvalues.len());
        let sorted_values = values.select(Axis(0), &order);
        let sorted_ntc = is_ntc.select(Axis(0), &order);
        let sorted_fdr = Self::empirical_fdr(&sorted_ntc);
        let threshold = Self::threshold(&sorted_values, &sorted_fdr, self.alpha);
        let unsorted_fdr = sorted_fdr.select(Axis(0), &reorder);
        FdrResult::new(unsorted_fdr, threshold)
    }

    /// Create a mask for the non-target controls
    fn ntc_mask(ntc_indices: &[usize], n_genes: usize) -> Array1<f64> {
        let mut mask = Array1::zeros(n_genes);
        for idx in ntc_indices {
            mask[*idx] = 1.0;
        }
        mask
    }

    /// Calculate the empirical FDR
    fn empirical_fdr(sorted_ntc: &Array1<f64>) -> Array1<f64> {
        let mut ntc_count = 0;
        sorted_ntc
            .iter()
            .enumerate()
            .map(|(idx, is_ntc)| {
                if *is_ntc == 1.0 {
                    ntc_count += 1;
                }
                ntc_count as f64 / (idx + 1) as f64
            })
            .collect()
    }

    /// Calculate the p-value threshold
    fn threshold(values: &Array1<f64>, fdr: &Array1<f64>, alpha: f64) -> f64 {
        let fdr_pval = fdr
            .iter()
            .zip(values.iter())
            .take_while(|(fdr, _value)| *fdr <= &alpha)
            .reduce(|_x, y| y);
        if let Some(fp) = fdr_pval {
            *fp.1
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod testing {
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
        let logfc = array![0.1, 0.2, 0.3];
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
}
