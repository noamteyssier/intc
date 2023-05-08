use crate::{
    encode::EncodeIndex,
    rank_test::{pseudo_rank_test, rank_test},
    result::IncResult,
    utils::{build_pseudo_names, reconstruct_names, select_values, validate_token, diagonal_product, aggregate_fold_changes}, mwu::Alternative,
};
use anyhow::Result;
use ndarray::Array1;

#[derive(Debug)]
pub struct Inc<'a> {
    pvalues: &'a Array1<f64>,
    logfc: &'a Array1<f64>,
    genes: &'a [String],
    token: &'a str,
    n_pseudo: usize,
    s_pseudo: usize,
    alpha: f64,
    alternative: Alternative,
    continuity: bool,
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
        alpha: f64,
        alternative: Alternative,
        continuity: bool,
        seed: Option<u64>,
    ) -> Inc<'a> {
        Inc {
            pvalues,
            logfc,
            genes,
            token,
            n_pseudo,
            s_pseudo,
            alpha,
            alternative,
            continuity,
            seed: seed.unwrap_or(0),
        }
    }

    pub fn fit(&self) -> Result<IncResult> {
        let encoding = EncodeIndex::new(self.genes);
        let product = diagonal_product(self.logfc, self.pvalues);
        let ntc_index = validate_token(&encoding.map, self.token)?;
        let ntc_pvalues = select_values(ntc_index, encoding.encoding(), self.pvalues);
        let ntc_logfcs = select_values(ntc_index, encoding.encoding(), self.logfc);
        let ntc_product = diagonal_product(&ntc_logfcs, &ntc_pvalues);
        let n_genes = encoding.map.len() - 1;

        let gene_fc_map = aggregate_fold_changes(
            self.genes,
            self.logfc,
            self.pvalues,
        );

        // run the rank test on all genes
        let (mwu_scores, mwu_pvalues) = rank_test(
            n_genes,
            ntc_index,
            encoding.encoding(),
            &product,
            &ntc_product,
            self.alternative,
            self.continuity,
        );

        // run the rank test on pseudo genes
        let (pseudo_scores, pseudo_pvalues, pseudo_logfc) =
            pseudo_rank_test(
                self.n_pseudo, 
                self.s_pseudo, 
                &ntc_pvalues, 
                &ntc_logfcs,
                self.alternative, 
                self.continuity,
                self.seed,
            );

        // reconstruct the gene names
        let gene_names = reconstruct_names(encoding.map(), ntc_index);
        let pseudo_names = build_pseudo_names(self.n_pseudo);

        // collect the gene fold changes
        let gene_logfc = gene_names
            .iter()
            .map(|gene| gene_fc_map.get(gene).unwrap())
            .copied()
            .collect::<Vec<f64>>();

        Ok(IncResult::new(
            gene_names,
            pseudo_names,
            mwu_scores,
            mwu_pvalues,
            gene_logfc,
            pseudo_scores,
            pseudo_pvalues,
            pseudo_logfc,
            self.alpha,
        ))
    }
}
