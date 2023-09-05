use csv::Error;
use rand::seq::SliceRandom;

use crate::tensor::Tensor;

use super::{add_test_train_split, Dataset};

pub struct IrisDataset {
    data: Tensor,
    labels: Tensor,
}

impl IrisDataset {
    pub fn from_csv(path: &str) -> Result<Self, Error> {
        let mut reader = csv::Reader::from_path(path)?;
        let mut records: Vec<_> = reader.records().flatten().collect();

        println!("MOVE SHUFFLE OUTSIDE THIS");
        records.shuffle(&mut rand::thread_rng());

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

        Ok(IrisDataset {
            data: attrib_tensor,
            labels: target_tensor,
        })
    }
}

impl Dataset for IrisDataset {
    fn get_data_labels(self) -> (Tensor, Tensor) {
        (self.data, self.labels)
    }

    add_test_train_split!();
}
