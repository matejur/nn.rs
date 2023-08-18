mod linear;
mod network;
mod tensor;

use linear::Linear;
use network::NeuralNetwork;
use tensor::{to_float, Tensor};

// TODO: Move this somewhere else
fn cost(out: &Tensor, target: &Tensor) -> f32 {
    if out.shape.len() != 2 {
        todo!("MSE: works only with 2D tensors");
    }

    let mut diff = out.sub_alloc(&target);
    diff.square();
    diff.sum() / out.shape[0] as f32
}

fn main() {
    let mut nn = NeuralNetwork::create(
        vec![
            Linear::new([2, 2]),
            Linear::new([2, 2]),
            Linear::new([2, 1]),
        ],
        cost,
    );

    let input = Tensor::from_array(vec![4, 2], &to_float(&[0, 0, 0, 1, 1, 0, 1, 1]));
    let target = Tensor::from_array(vec![4, 1], &to_float(&[0, 1, 1, 0]));

    let mut out = nn.forward(&input);
    for epoch in 0..500 {
        nn.gradient_finite_difference(&input, &target, 0.00003);
        nn.optimize(10.0);

        out = nn.forward(&input);
        println!("Epoch: {epoch} Cost: {}", cost(&out, &target));
    }
    println!("{out}");
}
