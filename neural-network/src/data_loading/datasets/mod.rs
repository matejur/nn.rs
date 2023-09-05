pub mod iris;
pub mod mnist_digits;

use crate::tensor::Tensor;

pub trait Dataset {
    fn get_data_labels(self) -> (Tensor, Tensor);
    fn train_test_split(self, train_ratio: f32) -> (Self, Self)
    where
        Self: Sized;
}

macro_rules! add_test_train_split {
    () => {
        fn train_test_split(self, train_ratio: f32) -> (Self, Self) {
            let (data, labels) = self.get_data_labels();
            let length = data.shape[0];
            let train_size = (train_ratio * length as f32) as usize;
            let test_size = length - train_size;

            let mut attribs_train: Vec<f32> = Vec::with_capacity(data.shape[1] * train_size);
            let mut labels_train: Vec<f32> = Vec::with_capacity(labels.shape[1] * train_size);

            for i in 0..train_size {
                attribs_train.extend(data.get_row(i).unwrap());
                labels_train.extend(labels.get_row(i).unwrap());
            }

            let mut attribs_test: Vec<f32> = Vec::with_capacity(data.shape[1] * test_size);
            let mut labels_test: Vec<f32> = Vec::with_capacity(labels.shape[1] * test_size);

            for i in train_size..length {
                attribs_test.extend(data.get_row(i).unwrap());
                labels_test.extend(labels.get_row(i).unwrap());
            }

            let attribs_train = Tensor::from_array(vec![train_size, data.shape[1]], &attribs_train);
            let labels_train = Tensor::from_array(vec![train_size, labels.shape[1]], &labels_train);
            let attribs_test = Tensor::from_array(vec![test_size, data.shape[1]], &attribs_test);
            let labels_test = Tensor::from_array(vec![test_size, labels.shape[1]], &labels_test);

            (
                Self {
                    data: attribs_train,
                    labels: labels_train,
                },
                Self {
                    data: attribs_test,
                    labels: labels_test,
                },
            )
        }
    };
}

pub(crate) use add_test_train_split;
