use ndarray::Array1;
use crate::fdr::{FdrResult, Fdr};

#[derive(Debug)]
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
        let ntc_indices = (0..n_pseudo).collect::<Vec<usize>>();
        let fdr = Fdr::new(&u_pvalues, &ntc_indices, alpha).fit();
        Self {
            genes,
            u_scores,
            u_pvalues,
            fdr,
        }
    }

    pub fn genes(&self) -> &[String] {
        &self.genes
    }

    pub fn u_scores(&self) -> &Array1<f64> {
        &self.u_scores
    }

    pub fn u_pvalues(&self) -> &Array1<f64> {
        &self.u_pvalues
    }

    pub fn fdr(&self) -> &Array1<f64> {
        self.fdr.fdr()
    }

    pub fn threshold(&self) -> f64 {
        self.fdr.threshold()
    }
}
