use nn::cost::CostFunction;
use nn::linear::{Activation, Linear};
use nn::network::NeuralNetwork;
use nn::tensor::Tensor;
use rand::seq::SliceRandom;

fn main() {
    let nn_finite_diff = NeuralNetwork::create(
        vec![
            Linear::new([4, 8], Activation::ReLU),
            Linear::new([8, 3], Activation::SoftmaxCrossEntropy),
        ],
        CostFunction::CrossEntropy,
    );

    let mut nn_backprop = NeuralNetwork::create(
        vec![
            Linear::new([4, 8], Activation::LeakyReLU(0.1)),
            Linear::new([8, 3], Activation::SoftmaxCrossEntropy),
        ],
        CostFunction::CrossEntropy,
    );

    let mut reader = csv::Reader::from_path("./datasets/iris.csv").unwrap();
    let mut records: Vec<_> = reader.records().flat_map(|f| f).collect();
    records.shuffle(&mut rand::thread_rng());

    const BATCH_SIZE: usize = 10;
    const EPOCHS: usize = 20000;

    let mut batches: Vec<(Tensor, Tensor)> = Vec::new();
    for batch in records.chunks_exact(BATCH_SIZE) {
        let mut attribs: Vec<f32> = Vec::new();
        let mut targets: Vec<f32> = Vec::new();

        for sample in batch {
            attribs.push(sample[0].parse::<f32>().unwrap() / 10.0);
            attribs.push(sample[1].parse::<f32>().unwrap() / 10.0);
            attribs.push(sample[2].parse::<f32>().unwrap() / 10.0);
            attribs.push(sample[3].parse::<f32>().unwrap() / 10.0);

            match &sample[4] {
                "Setosa" => targets.extend([1.0, 0.0, 0.0].iter()),
                "Versicolor" => targets.extend([0.0, 1.0, 0.0].iter()),
                "Virginica" => targets.extend([0.0, 0.0, 1.0].iter()),
                _ => panic!("Invalid plant name"),
            }
        }

        let attrib_tensor = Tensor::from_array(vec![BATCH_SIZE, 4], &attribs);
        let target_tensor = Tensor::from_array(vec![BATCH_SIZE, 3], &targets);
        batches.push((attrib_tensor, target_tensor));
    }

    for epoch in 0..EPOCHS {
        let mut cost = 0.0;
        for i in 0..batches.len() - 1 {
            let input = &batches[i].0;
            let target = &batches[i].1;

            cost += nn_backprop.train(&input, &target, 0.001);

            if cost.is_nan() {
                panic!();
            }
        }

        cost /= (batches.len() - 1) as f32;

        println!("Epoch: {epoch} cost: {cost}");
    }

    let out = nn_backprop.predict(&batches[batches.len() - 1].0);
    println!("Output: {:?}", out.argmax());
    println!("Target: {:?}", batches[batches.len() - 1].1.argmax());
}
