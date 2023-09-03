use nn::cost::CostFunction;
use nn::data_loader::DataLoader;
use nn::linear::{Activation, Linear};
use nn::network::NeuralNetwork;
use nn::tensor::Tensor;
use rand::seq::SliceRandom;

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

    let mut reader = csv::Reader::from_path("./datasets/iris.csv").unwrap();
    let mut records: Vec<_> = reader.records().flat_map(|f| f).collect();
    records.shuffle(&mut rand::thread_rng());

    let mut attribs: Vec<f32> = Vec::new();
    let mut targets: Vec<f32> = Vec::new();

    for sample in records.iter().skip(BATCH_SIZE) {
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

    let attrib_tensor = Tensor::from_array(vec![records.len() - BATCH_SIZE, 4], &attribs);
    let target_tensor = Tensor::from_array(vec![records.len() - BATCH_SIZE, 3], &targets);

    let data_loader = DataLoader::new(attrib_tensor, target_tensor, BATCH_SIZE, true)
        .expect("Data loader creation error");

    for epoch in 0..EPOCHS {
        let mut cost_backprop = 0.0;
        let mut cost_finite = 0.0;
        for (data, labels) in data_loader.iter() {
            cost_backprop += nn_backprop.train(&data, &labels, 0.01);

            nn_finite_diff.gradient_finite_difference(&data, &labels, 1e-4);
            nn_finite_diff.optimize(0.01);
            cost_finite += nn_finite_diff.cost_input_target(&data, &labels);
        }

        cost_backprop /= (records.len() / BATCH_SIZE) as f32;
        cost_finite /= (records.len() / BATCH_SIZE) as f32;

        println!("Epoch: {epoch} cost_backprop: {cost_backprop} cost_finite: {cost_finite}");
    }

    let mut attribs: Vec<f32> = Vec::new();
    let mut targets: Vec<f32> = Vec::new();

    for sample in records.iter().take(BATCH_SIZE) {
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

    let test_data = Tensor::from_array(vec![BATCH_SIZE, 4], &attribs);
    let test_labels = Tensor::from_array(vec![BATCH_SIZE, 3], &targets);

    let out = nn_backprop.predict(&test_data);
    println!("Output backprop: {:?}", out.argmax());
    let out = nn_finite_diff.predict(&test_data);
    println!("Output finite:   {:?}", out.argmax());
    println!("Target:          {:?}", test_labels.argmax());
}
