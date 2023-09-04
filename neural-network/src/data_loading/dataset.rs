use std::{
    io::{Cursor, Read},
    path::Path,
};

use csv::Error;
use rand::seq::SliceRandom;

use crate::tensor::Tensor;

pub struct Dataset {
    pub data: Tensor,
    pub labels: Tensor,
}

fn mnist_bytes_to_tensor(bytes: &[u8]) -> Tensor {
    let mut cursor = Cursor::new(bytes);

    let mut shape = Vec::new();
    let mut data = Vec::new();

    let mut i32_buffer = [0; 4];

    cursor
        .read_exact(&mut i32_buffer)
        .expect("Wrong MNIST format");
    let magic_number = i32::from_be_bytes(i32_buffer);

    match magic_number {
        2049 => {
            cursor
                .read_exact(&mut i32_buffer)
                .expect("Wrong MNIST format");
            shape.push(i32::from_be_bytes(i32_buffer) as usize);
            shape.push(10);
        }
        2051 => {
            cursor
                .read_exact(&mut i32_buffer)
                .expect("Wrong MNIST format");
            shape.push(i32::from_be_bytes(i32_buffer) as usize);
            cursor
                .read_exact(&mut i32_buffer)
                .expect("Wrong MNIST format");
            shape.push(i32::from_be_bytes(i32_buffer) as usize);
            cursor
                .read_exact(&mut i32_buffer)
                .expect("Wrong MNIST format");
            shape.push(i32::from_be_bytes(i32_buffer) as usize);
        }
        _ => panic!("Invalid magic number for MNIST dataset: {}", magic_number),
    }

    cursor.read_to_end(&mut data).expect("Wrong MNIST format");

    let mut tensor_data;
    match magic_number {
        2049 => {
            tensor_data = vec![0.0; data.len() * 10];

            for (i, label) in data.iter().enumerate() {
                let index = i * 10 + *label as usize;
                tensor_data[index] = 1 as f32;
            }
        }
        2051 => {
            tensor_data = data.iter().map(|x| *x as f32 / 255.0).collect::<Vec<_>>();
        }
        _ => panic!("Invalid magic number for MNIST dataset: {}", magic_number),
    }

    Tensor::from_array(shape, &tensor_data)
}

impl Dataset {
    pub fn iris_from_csv(path: &str) -> Result<Self, Error> {
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

        Ok(Dataset {
            data: attrib_tensor,
            labels: target_tensor,
        })
    }

    // Download the dataset from http://yann.lecun.com/exdb/mnist/
    pub fn mnist_from_directory(path: &str) -> Result<(Self, Self), Error> {
        let train_images_bytes = std::fs::read(Path::new(path).join("train-images-idx3-ubyte"))?;
        let train_labels_bytes = std::fs::read(Path::new(path).join("train-labels-idx1-ubyte"))?;

        let test_images_bytes = std::fs::read(Path::new(path).join("t10k-images-idx3-ubyte"))?;
        let test_labels_bytes = std::fs::read(Path::new(path).join("t10k-labels-idx1-ubyte"))?;

        let mut train_images = mnist_bytes_to_tensor(&train_images_bytes);
        train_images.reshape(vec![
            train_images.shape[0],
            train_images.shape[1] * train_images.shape[2],
        ]);
        let train_labels = mnist_bytes_to_tensor(&train_labels_bytes);

        let mut test_images = mnist_bytes_to_tensor(&test_images_bytes);
        test_images.reshape(vec![
            test_images.shape[0],
            test_images.shape[1] * test_images.shape[2],
        ]);
        let test_labels = mnist_bytes_to_tensor(&test_labels_bytes);

        Ok((
            Dataset {
                data: train_images,
                labels: train_labels,
            },
            Dataset {
                data: test_images,
                labels: test_labels,
            },
        ))
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
