use nn::{
    cost::CostFunction,
    data_loading::{
        data_loader::DataLoader,
        datasets::mnist_digits::{MnistDataset, MnistSplit},
    },
    linear::{Activation, Linear},
    network::NeuralNetwork,
};

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 20;

fn main() {
    let train_dataset = MnistDataset::from_directory("./datasets/MNIST_digits", MnistSplit::Train)
        .expect("Iris dataset creation failed!");

    let test_dataset = MnistDataset::from_directory("./datasets/MNIST_digits", MnistSplit::Test)
        .expect("Iris dataset creation failed!");

    let train_loader = DataLoader::new(train_dataset, BATCH_SIZE, true);
    let test_loader = DataLoader::new(test_dataset, 1, true);

    let mut nn = NeuralNetwork::create(
        vec![
            Linear::new([784, 32], Activation::ReLU),
            Linear::new([32, 10], Activation::SoftmaxCrossEntropy),
        ],
        CostFunction::CrossEntropy,
    );

    for epoch in 0..EPOCHS {
        let mut cost = 0.0;
        for (data, labels) in train_loader.iter() {
            cost += nn.train(&data, &labels, 0.0001);
        }

        cost /= (train_loader.len() / BATCH_SIZE) as f32;

        println!("Epoch: {epoch} cost: {cost}");
    }

    let mut iter = test_loader.iter();
    for _ in 0..10 {
        let (data, label) = iter.next().unwrap();
        MnistDataset::display_number(&data.get_row(0).unwrap());

        let out = nn.predict(&data);
        println!("Output: {:?}", out.argmax());
        println!("Target: {:?}", label.argmax());
    }
}
