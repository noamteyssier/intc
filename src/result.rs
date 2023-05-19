use crate::fdr::{Direction, Fdr, FdrResult};
use ndarray::{Array1, Array2};

#[derive(Debug)]
/// Result struct for Mann-Whitney U test and Empirical FDR
pub struct IncResult {
    genes: Vec<String>,
    u_scores: Array1<f64>,
    u_pvalues: Array1<f64>,
    logfc: Array1<f64>,
    fdr: FdrResult,
}
impl IncResult {
    pub fn new(
        genes: Vec<String>,
        u_scores: Array1<f64>,
        u_pvalues: Array1<f64>,
        logfc: Array1<f64>,
        matrix_pvalues: Array2<f64>,
        matrix_logfc: Array2<f64>,
        alpha: f64,
        use_product: Option<Direction>,
    ) -> Self {
        let fdr = Fdr::new(
            &u_pvalues,
            &logfc,
            &matrix_pvalues,
            &matrix_logfc,
            alpha,
            use_product,
        )
        .fit();
        Self {
            genes,
            u_scores,
            u_pvalues,
            logfc,
            fdr,
        }
    }

    /// Get the genes
    pub fn genes(&self) -> &[String] {
        &self.genes
    }

    /// Get the U scores
    pub fn u_scores(&self) -> &Array1<f64> {
        &self.u_scores
    }

    /// Get the U p-values
    pub fn u_pvalues(&self) -> &Array1<f64> {
        &self.u_pvalues
    }

    /// Get the FDR values
    pub fn fdr(&self) -> &Array1<f64> {
        self.fdr.fdr()
    }

    /// Get the log fold changes
    pub fn logfc(&self) -> &Array1<f64> {
        &self.logfc
    }

    /// Get the p-value threshold
    pub fn threshold(&self) -> f64 {
        self.fdr.threshold()
    }
}

#[cfg(test)]
mod testing {
    use super::*;

    #[test]
    fn test_inc_result() {
        let genes = vec!["a", "b", "c", "d", "e", "f"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let u_scores = Array1::linspace(0., 1., 6);
        let u_pvalues = Array1::linspace(0., 1., 6);
        let logfc = Array1::linspace(0., 1., 6);
        let matrix_pvalues = Array2::zeros((6, 6));
        let matrix_logfc = Array2::zeros((6, 6));
        let alpha = 0.05;
        let use_product = None;
        let result = IncResult::new(
            genes,
            u_scores.clone(),
            u_pvalues.clone(),
            logfc.clone(),
            matrix_pvalues,
            matrix_logfc,
            alpha,
            use_product,
        );
        assert_eq!(result.genes(), &["a", "b", "c", "d", "e", "f"]);
        assert_eq!(result.u_scores(), &u_scores);
        assert_eq!(result.u_pvalues(), &u_pvalues);
        assert_eq!(result.logfc(), &logfc);
        assert_eq!(result.fdr().len(), 6);
        assert_eq!(result.threshold(), 0.);
    }
}
