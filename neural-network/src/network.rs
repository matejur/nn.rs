use std::cell::Ref;

use crate::{cost::CostFunction, linear::Linear, tensor::Tensor};

pub struct NeuralNetwork {
    layers: Vec<Linear>,
    cost_function: CostFunction,
}

impl NeuralNetwork {
    pub fn create(layers: Vec<Linear>, cost_function: CostFunction) -> Self {
        Self {
            layers,
            cost_function,
        }
    }

    pub fn predict(&self, x: &Tensor) -> Ref<Tensor> {
        let mut x = self.layers[0].forward(x);

        for layer in &self.layers[1..] {
            x = layer.forward(&x);
        }

        x
    }

    pub fn backward(&mut self, input: &Tensor, mut gradient: Tensor) {
        for layer_index in (1..self.layers.len()).rev() {
            let (before, after) = self.layers.split_at_mut(layer_index);

            let layer_input = before[layer_index - 1].get_activations_reference();
            gradient = after[0].backward(&layer_input, gradient);
        }

        self.layers[0].backward(input, gradient);
    }

    pub fn train(&mut self, input: &Tensor, target: &Tensor, lr: f32) -> f32 {
        // TODO: look at these clones??
        let out = self.predict(input).clone();
        let gradient = match self.cost_function {
            CostFunction::CrossEntropy => out.sub_alloc(target),
            CostFunction::MeanSquaredError => out.sub_alloc(target),
        };

        self.backward(input, gradient);
        self.optimize(lr);

        self.cost_function.compute(&out, target)
    }

    pub fn optimize(&mut self, lr: f32) {
        for i in 0..self.layers.len() {
            self.layers[i].optimize(lr);
        }
    }

    pub fn print_layers(&self) {
        for layer in &self.layers {
            println!("Weights:\n{} Biases:\n{}", layer.weights, layer.biases);
        }
    }

    pub fn print_gradients(&self) {
        for layer in &self.layers {
            println!(
                "Weights grad:\n{} Biases grad:\n{}",
                layer.weights_grad, layer.biases_grad
            );
        }
    }

    pub fn get_layers(&self) -> &Vec<Linear> {
        &self.layers
    }

    pub fn cost_input_target(&self, x: &Tensor, target: &Tensor) -> f32 {
        let out = self.predict(x);
        self.cost_function.compute(&out, target)
    }

    // This function is a mess, was fighting with the borrow checker quite a lot...
    pub fn gradient_finite_difference(&mut self, x: &Tensor, target: &Tensor, epsilon: f32) {
        let initial_cost = self.cost_input_target(x, target);

        for layer_index in 0..self.layers.len() {
            // Weights
            for i in 0..self.layers[layer_index].weights.elems.len() {
                let prev_weight = self.layers[layer_index].weights.elems[i];
                self.layers[layer_index].weights.elems[i] += epsilon;
                let new_cost = self.cost_input_target(x, target);
                self.layers[layer_index].weights_grad.elems[i] =
                    (new_cost - initial_cost) / epsilon;
                self.layers[layer_index].weights.elems[i] = prev_weight;
            }

            // Biases
            for i in 0..self.layers[layer_index].biases.elems.len() {
                let prev_bias = self.layers[layer_index].biases.elems[i];
                self.layers[layer_index].biases.elems[i] += epsilon;
                let new_cost = self.cost_input_target(x, target);
                self.layers[layer_index].biases_grad.elems[i] = (new_cost - initial_cost) / epsilon;
                self.layers[layer_index].biases.elems[i] = prev_bias;
            }
        }
    }
}
