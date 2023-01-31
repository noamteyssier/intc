use hashbrown::HashMap;

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

    pub fn encoding(&self) -> &[usize] {
        &self.encoding
    }

    pub fn map(&self) -> &HashMap<usize, &str> {
        &self.map
    }
}

#[cfg(test)]
mod testing {
    use super::EncodeIndex;
    use hashbrown::HashMap;

    #[test]
    fn test_encode() {
        let genes = vec!["a", "b", "c", "a", "b", "c", "a", "b", "c"]
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>();
        let encode = EncodeIndex::new(&genes);
        assert_eq!(encode.encoding(), &[0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(
            encode.map(),
            &[(0, "a"), (1, "b"), (2, "c")]
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect::<HashMap<usize, &str>>()
        );
    }
}
