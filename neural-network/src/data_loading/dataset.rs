use csv::Error;
use rand::seq::SliceRandom;

use crate::tensor::Tensor;

pub struct Dataset {
    pub data: Tensor,
    pub labels: Tensor,
}

impl Dataset {
    pub fn iris_from_csv(path: &str, shuffle: bool) -> Result<Self, Error> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut records: Vec<_> = reader.records().flatten().collect();

        if shuffle {
            records.shuffle(&mut rand::thread_rng());
        }

        let mut attribs: Vec<f32> = Vec::new();
        let mut targets: Vec<f32> = Vec::new();

        for sample in records.iter() {
            attribs.push(sample[0].parse::<f32>().unwrap());
            attribs.push(sample[1].parse::<f32>().unwrap());
            attribs.push(sample[2].parse::<f32>().unwrap());
            attribs.push(sample[3].parse::<f32>().unwrap());

            match &sample[4] {
                "Setosa" => targets.extend([1.0, 0.0, 0.0].iter()),
                "Versicolor" => targets.extend([0.0, 1.0, 0.0].iter()),
                "Virginica" => targets.extend([0.0, 0.0, 1.0].iter()),
                _ => panic!("Invalid plant name"),
            }
        }

        let attrib_tensor = Tensor::from_array(vec![records.len(), 4], &attribs);
        let target_tensor = Tensor::from_array(vec![records.len(), 3], &targets);

        Ok(Dataset {
            data: attrib_tensor,
            labels: target_tensor,
        })
    }

    pub fn train_test_split(self, train_ratio: f32) -> (Dataset, Dataset) {
        let length = self.data.shape[0];
        let train_size = (train_ratio * length as f32) as usize;
        let test_size = length - train_size;

        let mut attribs_train: Vec<f32> = Vec::with_capacity(self.data.shape[1] * train_size);
        let mut labels_train: Vec<f32> = Vec::with_capacity(self.labels.shape[1] * train_size);

        for i in 0..train_size {
            attribs_train.extend(self.data.get_row(i).unwrap());
            labels_train.extend(self.labels.get_row(i).unwrap());
        }

        let mut attribs_test: Vec<f32> = Vec::with_capacity(self.data.shape[1] * test_size);
        let mut labels_test: Vec<f32> = Vec::with_capacity(self.labels.shape[1] * test_size);

        for i in train_size..length {
            attribs_test.extend(self.data.get_row(i).unwrap());
            labels_test.extend(self.labels.get_row(i).unwrap());
        }

        let attribs_train =
            Tensor::from_array(vec![train_size, self.data.shape[1]], &attribs_train);
        let labels_train =
            Tensor::from_array(vec![train_size, self.labels.shape[1]], &labels_train);
        let attribs_test = Tensor::from_array(vec![test_size, self.data.shape[1]], &attribs_test);
        let labels_test = Tensor::from_array(vec![test_size, self.labels.shape[1]], &labels_test);

        (
            Dataset {
                data: attribs_train,
                labels: labels_train,
            },
            Dataset {
                data: attribs_test,
                labels: labels_test,
            },
        )
    }
}
