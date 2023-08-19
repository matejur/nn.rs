mod linear;
mod network;
mod tensor;

use linear::Linear;
use network::NeuralNetwork;
use rand::seq::SliceRandom;
use tensor::Tensor;

// TODO: Move this somewhere else
fn mse(out: &Tensor, target: &Tensor) -> f32 {
    let mut diff = out.sub_alloc(&target);
    diff.square();
    diff.sum() / out.shape[0] as f32
}

fn cross_entropy(out: &Tensor, target: &Tensor) -> f32 {
    let mut loss = 0.0;
    for i in 0..out.elems.len() {
        let e1 = target.elems[i].clamp(1e-5, 1.0 - 1e-5);
        let e2 = out.elems[i].clamp(1e-5, 1.0 - 1e-5);
        loss += e1 * e2.ln();
        loss += (1.0 - e1) * (1.0 - e2).ln();
    }

    -loss / out.elems.len() as f32
}

fn main() {
    let mut nn = NeuralNetwork::create(
        vec![Linear::new([4, 8]), Linear::new([8, 3])],
        cross_entropy,
    );

    let mut reader = csv::Reader::from_path("./datasets/iris.csv").unwrap();
    let mut records: Vec<_> = reader.records().flat_map(|f| f).collect();
    records.shuffle(&mut rand::thread_rng());

    const BATCH_SIZE: usize = 10;
    const EPOCHS: usize = 2000;

    let mut batches: Vec<(Tensor, Tensor)> = Vec::new();
    for batch in records.chunks_exact(BATCH_SIZE) {
        let mut attribs: Vec<f32> = Vec::new();
        let mut targets: Vec<f32> = Vec::new();

        for sample in batch {
            attribs.push(sample[0].parse().unwrap());
            attribs.push(sample[1].parse().unwrap());
            attribs.push(sample[2].parse().unwrap());
            attribs.push(sample[3].parse().unwrap());

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

            nn.gradient_finite_difference(&input, &target, 1e-6);
            nn.optimize(0.01);

            let out = nn.forward(&input);
            cost += cross_entropy(&out, &target);
        }

        cost /= (batches.len() - 1) as f32;

        println!("Epoch: {epoch} Cost: {cost}",);
        if cost < 1e-2 {
            break;
        }
    }

    let out = nn.forward(&batches[batches.len() - 1].0);
    println!("Output: {:?}", out.argmax());
    println!("Target: {:?}", batches[batches.len() - 1].1.argmax());
}
