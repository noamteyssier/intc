use intc::inc::Inc;
use ndarray::Array1;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

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
    let pvalues = Array1::random(m, Uniform::new(1e-8, 1.0));
    let token = "non-targeting";
    let inc = Inc::new(&pvalues, &genes, token, 100, 5);
    let res = inc.fit().unwrap();
}
