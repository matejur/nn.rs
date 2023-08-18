use crate::{linear::Linear, tensor::Tensor};

pub struct NeuralNetwork {
    layers: Vec<Linear>,
    cost_function: fn(&Tensor, &Tensor) -> f32,
}

impl NeuralNetwork {
    pub fn create(layers: Vec<Linear>, cost_function: fn(&Tensor, &Tensor) -> f32) -> Self {
        Self {
            layers,
            cost_function,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = self.layers[0].forward(x);
        x.sigmoid();

        for layer in &self.layers[1..] {
            x = layer.forward(&x);
            x.sigmoid();
        }

        x
    }

    pub fn optimize(&mut self, lr: f32) {
        for i in 0..self.layers.len() {
            self.layers[i].optimize(lr);
        }
    }

    pub fn cost_input_target(&self, x: &Tensor, target: &Tensor) -> f32 {
        let out = self.forward(x);
        (self.cost_function)(&out, &target)
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
