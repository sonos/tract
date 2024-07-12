pub struct IndexIterator {
    shape: Vec<usize>,
    current_index: Vec<usize>,
    done: bool,
}

impl IndexIterator {
    pub fn new(shape: &[usize]) -> Self {
        let current_index = vec![0; shape.len()];
        Self { shape: shape.to_vec(), current_index, done: false }
    }
}

impl Iterator for IndexIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current_index.clone();

        for i in (0..self.shape.len()).rev() {
            if self.current_index[i] + 1 < self.shape[i] {
                self.current_index[i] += 1;
                // Reset all indices to the right of i to 0
                for j in i + 1..self.shape.len() {
                    self.current_index[j] = 0;
                }
                return Some(result);
            }
        }

        self.done = true;
        Some(result)
    }
}

pub fn iter_indices(shape: &[usize]) -> IndexIterator {
    IndexIterator::new(shape)
}

#[cfg(test)]
mod test {
    use super::iter_indices;
    #[test]
    fn test_single_element() {
        let shape = vec![1, 1, 1];
        let expected_indices = vec![vec![0, 0, 0]];
        let iter = iter_indices(&shape);
        let result: Vec<Vec<usize>> = iter.collect();
        assert_eq!(result, expected_indices);
    }

    #[test]
    fn test_3x1x1() {
        let shape = vec![3, 1, 1];
        let expected_indices = vec![vec![0, 0, 0], vec![1, 0, 0], vec![2, 0, 0]];
        let iter = iter_indices(&shape);
        let result: Vec<Vec<usize>> = iter.collect();
        assert_eq!(result, expected_indices);
    }

    #[test]
    fn test_2x2x1() {
        let shape = vec![2, 2, 1];
        let expected_indices = vec![vec![0, 0, 0], vec![0, 1, 0], vec![1, 0, 0], vec![1, 1, 0]];
        let iter = iter_indices(&shape);
        let result: Vec<Vec<usize>> = iter.collect();
        assert_eq!(result, expected_indices);
    }
}
