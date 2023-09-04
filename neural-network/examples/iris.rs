use nn::cost::CostFunction;
use nn::data_loading::data_loader::DataLoader;
use nn::data_loading::dataset::Dataset;
use nn::linear::{Activation, Linear};
use nn::network::NeuralNetwork;

fn main() {
    let mut nn_finite_diff = NeuralNetwork::create(
        vec![
            Linear::new([4, 8], Activation::ReLU),
            Linear::new([8, 3], Activation::SoftmaxCrossEntropy),
        ],
        CostFunction::CrossEntropy,
    );

    let mut nn_backprop = NeuralNetwork::create(
        vec![
            Linear::new([4, 8], Activation::ReLU),
            Linear::new([8, 3], Activation::SoftmaxCrossEntropy),
        ],
        CostFunction::CrossEntropy,
    );

    const BATCH_SIZE: usize = 10;
    const EPOCHS: usize = 1000;

    let dataset =
        Dataset::iris_from_csv("./datasets/iris.csv", true).expect("Iris dataset creation failed!");
    let (train_dataset, test_dataset) = dataset.train_test_split(0.9);

    let train_loader = DataLoader::new(train_dataset, BATCH_SIZE, false);

    let test_batch = test_dataset.data.shape[0];
    let test_loader = DataLoader::new(test_dataset, test_batch, false);

    for epoch in 0..EPOCHS {
        let mut cost_backprop = 0.0;
        let mut cost_finite = 0.0;
        for (data, labels) in train_loader.iter() {
            cost_backprop += nn_backprop.train(&data, &labels, 0.01);

            nn_finite_diff.gradient_finite_difference(&data, &labels, 1e-4);
            nn_finite_diff.optimize(0.01);
            cost_finite += nn_finite_diff.cost_input_target(&data, &labels);
        }

        cost_backprop /= (train_loader.len() / BATCH_SIZE) as f32;
        cost_finite /= (train_loader.len() / BATCH_SIZE) as f32;

        println!("Epoch: {epoch} cost_backprop: {cost_backprop} cost_finite: {cost_finite}");
    }

    for (data, labels) in test_loader.iter() {
        let out = nn_backprop.predict(&data);
        println!("Output backprop: {:?}", out.argmax());
        let out = nn_finite_diff.predict(&data);
        println!("Output finite:   {:?}", out.argmax());
        println!("Target:          {:?}", labels.argmax());
    }
}
