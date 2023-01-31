use crate::{
    encode::EncodeIndex,
    rank_test::{pseudo_rank_test, rank_test},
    result::IncResult,
    utils::{select_ranks, validate_token},
};
use anyhow::Result;
use hashbrown::HashMap;
use ndarray::Array1;

#[derive(Debug)]
pub struct Inc<'a> {
    pvalues: &'a Array1<f64>,
    genes: &'a [String],
    token: &'a str,
    n_pseudo: usize,
    s_pseudo: usize,
}

impl<'a> Inc<'a> {
    pub fn new(
        pvalues: &'a Array1<f64>,
        genes: &'a [String],
        token: &'a str,
        n_pseudo: usize,
        s_pseudo: usize,
    ) -> Inc<'a> {
        Inc {
            pvalues,
            genes,
            token,
            n_pseudo,
            s_pseudo,
        }
    }

    pub fn fit(&self) -> Result<IncResult> {
        let encoding = EncodeIndex::new(self.genes);
        let ntc_index = validate_token(&encoding.map, self.token)?;
        let ntc_values = select_ranks(ntc_index, encoding.encoding(), self.pvalues);
        let n_genes = encoding.map.len() - 1;

        // run the rank test on all genes
        let (mwu_scores, mwu_pvalues) = rank_test(
            n_genes,
            ntc_index,
            encoding.encoding(),
            self.pvalues,
            &ntc_values,
        );
        let (pseudo_scores, pseudo_pvalues) =
            pseudo_rank_test(self.n_pseudo, self.s_pseudo, &ntc_values);

        // reconstruct the gene names
        let gene_names = self.reconstruct_names(&encoding.map, ntc_index);
        let pseudo_names = self.build_pseudo_names(self.n_pseudo);

        Ok(IncResult::new(
            gene_names,
            pseudo_names,
            mwu_scores,
            mwu_pvalues,
            pseudo_scores,
            pseudo_pvalues,
        ))
    }

    fn reconstruct_names(&self, map: &HashMap<usize, &str>, ntc_index: usize) -> Vec<String> {
        (0..map.len())
            .filter(|x| *x != ntc_index)
            .map(|x| map.get(&x).unwrap().to_string())
            .collect()
    }

    fn build_pseudo_names(&self, n_pseudo: usize) -> Vec<String> {
        (0..n_pseudo).map(|x| format!("pseudogene-{}", x)).collect()
    }
}
