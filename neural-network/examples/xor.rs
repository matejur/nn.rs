use nn::cost::CostFunction;
use nn::linear::{Activation, Linear};
use nn::network::NeuralNetwork;
use nn::tensor::{to_float, Tensor};

fn main() {
    let mut nn = NeuralNetwork::create(
        vec![
            Linear::new([2, 2], Activation::Sigmoid),
            Linear::new([2, 1], Activation::Sigmoid),
        ],
        CostFunction::CrossEntropy,
    );

    let input = Tensor::from_array(vec![4, 2], &to_float(&[0, 0, 0, 1, 1, 0, 1, 1]));
    let target = Tensor::from_array(vec![4, 1], &to_float(&[0, 1, 1, 0]));

    for epoch in 0..2000 {
        nn.gradient_finite_difference(&input, &target, 1e-2);
        nn.optimize(1.0);

        println!(
            "Epoch: {epoch} Cost: {}",
            nn.cost_input_target(&input, &target)
        );
    }

    let out = nn.predict(&input);
    println!("{out}");
}
