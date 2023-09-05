use rand::seq::SliceRandom;

use crate::tensor::Tensor;

use super::datasets::Dataset;

pub struct DataLoader {
    data: Tensor,
    labels: Tensor,
    batch_size: usize,
    shuffle: bool,
}

#[derive(Debug)]
pub enum DataLoaderError {
    MissmatchedRowCount,
}

pub struct DataLoaderIter<'a> {
    data_loader: &'a DataLoader,
    indices: Vec<usize>,
    batch_size: usize,
    batch_index: usize,
}

impl DataLoader {
    pub fn new(dataset: impl Dataset, batch_size: usize, shuffle: bool) -> Self {
        let (data, labels) = dataset.get_data_labels();
        if data.shape[0] % batch_size != 0 {
            eprintln!("DATALOADER WARNING: The last {} rows will be dropped. Batch size does not evenly divide provided data TODO!", data.shape[0] % batch_size)
        }

        DataLoader {
            data,
            labels,
            batch_size,
            shuffle,
        }
    }

    pub fn iter(&self) -> DataLoaderIter {
        let mut indices: Vec<_> = (0..self.data.shape[0]).collect();

        if self.shuffle {
            indices.shuffle(&mut rand::thread_rng());
        }

        DataLoaderIter {
            data_loader: self,
            indices,
            batch_size: self.batch_size,
            batch_index: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.data.shape[0]
    }
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        let data = &self.data_loader.data;
        let labels = &self.data_loader.labels;
        let start_index = self.batch_index * self.batch_size;

        if start_index + self.batch_size > data.shape[0] {
            return None;
        }

        self.batch_index += 1;

        let mut data_batch = Vec::with_capacity(self.batch_size * data.shape[1]);
        let mut labels_batch = Vec::with_capacity(self.batch_size * labels.shape[1]);

        for i in 0..self.batch_size {
            let row_index = self.indices[start_index + i];
            data_batch.extend(data.get_row(row_index).unwrap());
            labels_batch.extend(labels.get_row(row_index).unwrap());
        }

        let data_batch = Tensor::from_array(vec![self.batch_size, data.shape[1]], &data_batch);
        let labels_batch =
            Tensor::from_array(vec![self.batch_size, labels.shape[1]], &labels_batch);

        Some((data_batch, labels_batch))
    }
}
