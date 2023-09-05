use std::{
    io::{Cursor, Read},
    path::Path,
};

use csv::Error;
use rand::seq::SliceRandom;

use crate::tensor::Tensor;

// pub struct Dataset {
//     pub data: Tensor,
//     pub labels: Tensor,
// }



// impl Dataset {
//     

//     // Download the dataset from http://yann.lecun.com/exdb/mnist/
//     

//     pub fn train_test_split(self, train_ratio: f32) -> (Dataset, Dataset) {
//         let length = self.data.shape[0];
//         let train_size = (train_ratio * length as f32) as usize;
//         let test_size = length - train_size;

//         let mut attribs_train: Vec<f32> = Vec::with_capacity(self.data.shape[1] * train_size);
//         let mut labels_train: Vec<f32> = Vec::with_capacity(self.labels.shape[1] * train_size);

//         for i in 0..train_size {
//             attribs_train.extend(self.data.get_row(i).unwrap());
//             labels_train.extend(self.labels.get_row(i).unwrap());
//         }

//         let mut attribs_test: Vec<f32> = Vec::with_capacity(self.data.shape[1] * test_size);
//         let mut labels_test: Vec<f32> = Vec::with_capacity(self.labels.shape[1] * test_size);

//         for i in train_size..length {
//             attribs_test.extend(self.data.get_row(i).unwrap());
//             labels_test.extend(self.labels.get_row(i).unwrap());
//         }

//         let attribs_train =
//             Tensor::from_array(vec![train_size, self.data.shape[1]], &attribs_train);
//         let labels_train =
//             Tensor::from_array(vec![train_size, self.labels.shape[1]], &labels_train);
//         let attribs_test = Tensor::from_array(vec![test_size, self.data.shape[1]], &attribs_test);
//         let labels_test = Tensor::from_array(vec![test_size, self.labels.shape[1]], &labels_test);

//         (
//             Dataset {
//                 data: attribs_train,
//                 labels: labels_train,
//             },
//             Dataset {
//                 data: attribs_test,
//                 labels: labels_test,
//             },
//         )
//     }
// }

