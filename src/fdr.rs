use ndarray::{Array1, Axis};
use crate::utils::argsort;

#[derive(Debug)]
pub struct FdrResult {
    fdr: Array1<f64>,
    threshold: f64,
}
impl FdrResult {
    pub fn new(fdr: Array1<f64>, threshold: f64) -> Self {
        Self { fdr, threshold }
    }
    pub fn fdr(&self) -> &Array1<f64> {
        &self.fdr
    }
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

#[derive(Debug)]
pub struct Fdr<'a> {
    pvalues: &'a Array1<f64>,
    ntc_indices: &'a [usize],
    alpha: f64,
}

impl<'a> Fdr<'a> {
    pub fn new(pvalues: &'a Array1<f64>, ntc_indices: &'a [usize], alpha: f64) -> Self {
        Self {
            pvalues,
            ntc_indices,
            alpha,
        }
    }
    
    pub fn fit(&self) -> FdrResult {
        let order = argsort(self.pvalues);
        let is_ntc = Self::ntc_mask(self.ntc_indices, self.pvalues.len());
        let sorted_pvalues = self.pvalues.select(Axis(0), &order);
        let sorted_ntc = is_ntc.select(Axis(0), &order);
        let sorted_fdr = Self::empirical_fdr(&sorted_ntc);
        let threshold = Self::threshold(&sorted_pvalues, &sorted_fdr, self.alpha);
        let unsorted_fdr = sorted_fdr.select(Axis(0), &order);
        FdrResult::new(unsorted_fdr, threshold)
    }
    
    fn ntc_mask(ntc_indices: &[usize], n_genes: usize) -> Array1<f64> {
        let mut mask = Array1::zeros(n_genes);
        for idx in ntc_indices {
            mask[*idx] = 1.0;
        }
        mask
    }

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

    fn threshold(pvalues: &Array1<f64>, fdr: &Array1<f64>, alpha: f64) -> f64 {
        let fdr_pval = fdr.iter().zip(pvalues.iter())
            .take_while(|(fdr, _pvalue)| *fdr <= &alpha)
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
    use ndarray::{array, Array1};
    use super::Fdr;

    #[test]
    fn test_fdr() {
        let pvalues = array![0.1, 0.2, 0.3];
        let ntc_indices = vec![1];
        let alpha = 0.1;
        let fdr = Fdr::new(&pvalues, &ntc_indices, alpha).fit();
        assert_eq!(fdr.fdr(), array![0.0, 0.5, 1./3.]);
    }

    #[test]
    fn test_fdr_unsorted() {
        let pvalues = array![0.2, 0.1, 0.3];
        let ntc_indices = vec![1];
        let alpha = 0.1;
        let fdr = Fdr::new(&pvalues, &ntc_indices, alpha).fit();
        assert_eq!(fdr.fdr(), array![0.5, 1.0, 1./3.]);
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
        let ntc_indices = vec![4, 6, 9];
        let alpha = 0.1;
        let fdr = Fdr::new(&pvalues, &ntc_indices, alpha).fit();
        assert_eq!(fdr.threshold(), 1./3.);
    }

}
