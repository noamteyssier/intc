use crate::{
    encode::EncodeIndex,
    fdr::Direction,
    mwu::Alternative,
    rank_test::{pseudo_rank_test_matrix, rank_test},
    result::IncResult,
    utils::{aggregate_fold_changes, reconstruct_names, select_values, validate_token},
};
use anyhow::Result;
use ndarray::Array1;

/// A struct for running the INC algorithm to aggregate p-values and fold changes
///
/// # Arguments
/// * `pvalues` - A vector of p-values
/// * `logfc` - A vector of log fold changes
/// * `genes` - A vector of gene names
/// * `token` - A token to identify the negative control genes
/// * `n_pseudo` - The number of pseudo genes to generate
/// * `s_pseudo` - The number of guides to sample for each pseudo gene
/// * `n_draws` - The number of draws to use for the INC algorithm
/// * `alpha` - The significance level threshold for the INC algorithm
/// * `alternative` - The alternative hypothesis for the Mann-Whitney U test
/// * `continuity` - Whether to use continuity correction in the Mann-Whitney U test
/// * `use_product` - Whether to use the product of p-values or fold changes
/// * `seed` - A seed for the random number generator
#[derive(Debug)]
pub struct Inc<'a> {
    pvalues: &'a Array1<f64>,
    logfc: &'a Array1<f64>,
    genes: &'a [String],
    token: &'a str,
    n_pseudo: usize,
    s_pseudo: usize,
    n_draws: usize,
    alpha: f64,
    alternative: Alternative,
    continuity: bool,
    use_product: Option<Direction>,
    seed: u64,
}

impl<'a> Inc<'a> {
    pub fn new(
        pvalues: &'a Array1<f64>,
        logfc: &'a Array1<f64>,
        genes: &'a [String],
        token: &'a str,
        n_pseudo: usize,
        s_pseudo: usize,
        n_draws: usize,
        alpha: f64,
        alternative: Alternative,
        continuity: bool,
        use_product: Option<Direction>,
        seed: Option<u64>,
    ) -> Inc<'a> {
        Inc {
            pvalues,
            logfc,
            genes,
            token,
            n_pseudo,
            s_pseudo,
            n_draws,
            alpha,
            alternative,
            continuity,
            use_product,
            seed: seed.unwrap_or(0),
        }
    }

    pub fn fit(&self) -> Result<IncResult> {
        let encoding = EncodeIndex::new(self.genes);
        let ntc_index = validate_token(&encoding.map, self.token)?;
        let ntc_pvalues = select_values(ntc_index, encoding.encoding(), self.pvalues);
        let ntc_logfcs = select_values(ntc_index, encoding.encoding(), self.logfc);
        let n_genes = encoding.map.len() - 1;

        let gene_fc_map = aggregate_fold_changes(self.genes, self.logfc);

        // run the rank test on all genes
        let (mwu_scores, mwu_pvalues) = rank_test(
            n_genes,
            ntc_index,
            encoding.encoding(),
            self.pvalues,
            &ntc_pvalues,
            Alternative::Less,
            self.continuity,
        );

        // run the rank test on multiple instantiations of pseudo genes
        let (matrix_pvalues, matrix_logfc) = pseudo_rank_test_matrix(
            self.n_pseudo,
            self.s_pseudo,
            self.n_draws,
            &ntc_pvalues,
            &ntc_logfcs,
            self.alternative,
            self.continuity,
            self.seed,
        );

        // reconstruct the gene names
        let gene_names = reconstruct_names(encoding.map(), ntc_index);

        // collect the gene fold changes
        let gene_logfc = gene_names
            .iter()
            .map(|gene| gene_fc_map.get(gene).unwrap())
            .copied()
            .collect::<Vec<f64>>();

        Ok(IncResult::new(
            gene_names,
            mwu_scores,
            mwu_pvalues,
            gene_logfc,
            matrix_pvalues,
            matrix_logfc,
            self.alpha,
            self.use_product,
        ))
    }
}
