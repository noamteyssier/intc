use hashbrown::HashMap;
use ndarray::Array1;

/// Converts a list of genes into a mapping of gene to index and a list of indices
#[derive(Debug)]
pub struct EncodeIndex<'a> {
    pub map: HashMap<usize, &'a str>,
    pub encoding: Vec<usize>,
}
impl<'a> EncodeIndex<'a> {
    pub fn new(genes: &'a [String]) -> Self {
        let mut total = 0usize;
        let mut map = HashMap::with_capacity(genes.len());
        let mut encoding = Vec::with_capacity(genes.len());
        for g in genes {
            if let Some(e) = map.get(g) {
                encoding.push(*e);
            } else {
                map.insert(g, total);
                encoding.push(total);
                total += 1;
            }
        }
        EncodeIndex {
            map: map.iter().map(|(k, v)| (*v, k.as_str())).collect(),
            encoding,
        }
    }

    pub fn encoding(&self) -> &[usize]{
        &self.encoding
    }

    pub fn map(&self) -> &HashMap<usize, &str> {
        &self.map
    }

}
