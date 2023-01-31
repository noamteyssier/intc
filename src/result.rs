use ndarray::Array1;

#[derive(Debug)]
pub struct IncResult {
    genes: Vec<String>,
    u_scores: Array1<f64>,
    u_pvalues: Array1<f64>,
}
impl IncResult {
    pub fn new(
        genes: Vec<String>,
        pseudo_genes: Vec<String>,
        gene_scores: Vec<f64>,
        gene_pvalues: Vec<f64>,
        pseudo_scores: Vec<f64>,
        pseudo_pvalues: Vec<f64>,
    ) -> Self {
        let genes = vec![genes, pseudo_genes].concat();
        let u_scores = vec![gene_scores, pseudo_scores].concat();
        let u_pvalues = vec![gene_pvalues, pseudo_pvalues].concat();
        Self {
            genes,
            u_scores: Array1::from_vec(u_scores),
            u_pvalues: Array1::from_vec(u_pvalues),
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
}
