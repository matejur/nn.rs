use nn::{
    cost::CostFunction,
    data_loading::{data_loader::DataLoader, dataset::Dataset},
    linear::{Activation, Linear},
    network::NeuralNetwork,
};

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 20;

fn display_number(data: &[f32]) {
    for i in 0..28 {
        for j in 0..28 {
            print!("{}", if data[i * 28 + j] == 0.0 { "  " } else { "##" });
        }
        println!();
    }
}
fn main() {
    let (train_dataset, test_dataset) = Dataset::mnist_from_directory("./datasets/MNIST_digits")
        .expect("Iris dataset creation failed!");

    let train_loader = DataLoader::new(train_dataset, BATCH_SIZE, true);
    let test_loader = DataLoader::new(test_dataset, 1, true);

    let mut nn = NeuralNetwork::create(
        vec![
            Linear::new([784, 32], Activation::ReLU),
            Linear::new([32, 32], Activation::ReLU),
            Linear::new([32, 10], Activation::SoftmaxCrossEntropy),
        ],
        CostFunction::CrossEntropy,
    );

    for epoch in 0..EPOCHS {
        let mut cost = 0.0;
        for (data, labels) in train_loader.iter() {
            cost += nn.train(&data, &labels, 0.0001);

            if cost.is_nan() {
                std::process::exit(0);
            }
        }

        cost /= (train_loader.len() / BATCH_SIZE) as f32;

        println!("Epoch: {epoch} cost: {cost}");
    }

    let mut iter = test_loader.iter();
    for _ in 0..10 {
        let (data, label) = iter.next().unwrap();
        display_number(&data.get_row(0).unwrap());

        let out = nn.predict(&data);
        println!("Output: {:?}", out.argmax());
        println!("Target: {:?}", label.argmax());
    }
}
