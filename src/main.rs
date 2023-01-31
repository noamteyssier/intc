use intc::inc::Inc;
use ndarray::Array1;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

fn main() {
    let m = 100;
    let genes = (0..m)
        .map(|x| {
            if x % 3 == 0 {
                "non-targeting".to_string()
            } else {
                format!("gene.{}", x % 5)
            }
        })
        .collect::<Vec<String>>();
    let pvalues = Array1::random(m, Uniform::new(0.1, 1.0));
    let token = "non-targeting";
    let alpha = 0.1;
    let n_pseudo = 100;
    let s_pseudo = 5;
    let inc = Inc::new(&pvalues, &genes, token, n_pseudo, s_pseudo, alpha);
    let _res = inc.fit().unwrap();
}
