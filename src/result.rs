use crate::fdr::{Fdr, FdrResult};
use ndarray::Array1;

#[derive(Debug)]
/// Result struct for Mann-Whitney U test and Empirical FDR
pub struct IncResult {
    genes: Vec<String>,
    u_scores: Array1<f64>,
    u_pvalues: Array1<f64>,
    fdr: FdrResult,
}
impl IncResult {
    pub fn new(
        genes: Vec<String>,
        pseudo_genes: Vec<String>,
        gene_scores: Vec<f64>,
        gene_pvalues: Vec<f64>,
        pseudo_scores: Vec<f64>,
        pseudo_pvalues: Vec<f64>,
        alpha: f64,
    ) -> Self {
        let n_pseudo = pseudo_genes.len();
        let genes = vec![genes, pseudo_genes].concat();
        let u_scores = Array1::from_vec(vec![gene_scores, pseudo_scores].concat());
        let u_pvalues = Array1::from_vec(vec![gene_pvalues, pseudo_pvalues].concat());
        let ntc_indices = Self::create_ntc_indices(n_pseudo, genes.len());
        let fdr = Fdr::new(&u_pvalues, &ntc_indices, alpha).fit();
        Self {
            genes,
            u_scores,
            u_pvalues,
            fdr,
        }
    }

    /// Create the indices for the non-targeting control genes by
    /// taking the indices of the last n pseudogenes
    fn create_ntc_indices(n_pseudo: usize, n_total: usize) -> Vec<usize> {
        (n_total-n_pseudo .. n_total).collect::<Vec<usize>>()
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

    /// Get the p-value threshold
    pub fn threshold(&self) -> f64 {
        self.fdr.threshold()
    }
}

#[cfg(test)]
mod testing {

    #[test]
    fn test_inc_result() {
        use super::*;
        let genes = vec!["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let pseudo_genes = vec!["k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let gene_scores = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let gene_pvalues = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let pseudo_scores = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let pseudo_pvalues = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let alpha = 0.05;
        let result = IncResult::new(
            genes,
            pseudo_genes,
            gene_scores,
            gene_pvalues,
            pseudo_scores,
            pseudo_pvalues,
            alpha,
        );
        assert_eq!(result.genes().len(), 20);
        assert_eq!(result.u_scores().len(), 20);
        assert_eq!(result.u_pvalues().len(), 20);
        assert_eq!(result.fdr().len(), 20);
        assert!(result.threshold() >= 0.);
    }

    #[test]
    fn test_ntc_indices() {
        use super::*;
        let n_pseudo = 10;
        let n_total = 20;
        let ntc_indices = IncResult::create_ntc_indices(n_pseudo, n_total);
        assert_eq!(ntc_indices.len(), 10);
        assert_eq!(
            ntc_indices,
            vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            );
    }
}
