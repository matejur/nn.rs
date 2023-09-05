use std::{
    io::{Cursor, Error, Read},
    path::Path,
};

use crate::tensor::Tensor;

use super::Dataset;

pub struct MnistDataset {
    data: Tensor,
    labels: Tensor,
}

pub enum MnistSplit {
    Train,
    Test,
}

impl MnistDataset {
    pub fn display_number(data: &[f32]) {
        for i in 0..28 {
            for j in 0..28 {
                print!("{}", if data[i * 28 + j] == 0.0 { "  " } else { "##" });
            }
            println!();
        }
    }
    pub fn from_directory(path: &str, split: MnistSplit) -> Result<Self, Error> {
        let (images_bytes, labels_bytes) = match split {
            MnistSplit::Train => (
                std::fs::read(Path::new(path).join("train-images-idx3-ubyte"))?,
                std::fs::read(Path::new(path).join("train-labels-idx1-ubyte"))?,
            ),
            MnistSplit::Test => (
                std::fs::read(Path::new(path).join("t10k-images-idx3-ubyte"))?,
                std::fs::read(Path::new(path).join("t10k-labels-idx1-ubyte"))?,
            ),
        };

        let mut images = mnist_bytes_to_tensor(&images_bytes);
        images.reshape(vec![images.shape[0], images.shape[1] * images.shape[2]]);
        let labels = mnist_bytes_to_tensor(&labels_bytes);

        Ok(Self {
            data: images,
            labels,
        })
    }
}

impl Dataset for MnistDataset {
    fn get_data_labels(self) -> (Tensor, Tensor) {
        (self.data, self.labels)
    }

    fn train_test_split(self, _train_ratio: f32) -> (Self, Self) {
        panic!("Use MnistType::Train and MnistType::Test when creating the dataset!");
    }
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
                tensor_data[index] = 1_f32;
            }
        }
        2051 => {
            tensor_data = data.iter().map(|x| *x as f32 / 255.0).collect::<Vec<_>>();
        }
        _ => panic!("Invalid magic number for MNIST dataset: {}", magic_number),
    }

    Tensor::from_array(shape, &tensor_data)
}
