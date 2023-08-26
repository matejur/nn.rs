use std::cell::{Ref, RefCell};

use crate::tensor::Tensor;

pub struct Linear {
    pub weights: Tensor,
    pub biases: Tensor,
    pub weights_grad: Tensor,
    pub biases_grad: Tensor,

    output: RefCell<Option<Tensor>>,
    activation: Activation,
}

pub enum Activation {
    ReLU,
    LeakyReLU(f32),
    Sigmoid,
    SoftmaxCrossEntropy,
}

impl Linear {
    pub fn new(shape: [usize; 2], activation: Activation) -> Self {
        let mut weights = Tensor::new(shape.to_vec());
        let mut biases = Tensor::new(vec![shape[1]]);
        let weights_grad = Tensor::new(shape.to_vec());
        let biases_grad = Tensor::new(vec![shape[1]]);

        weights.randomize();
        biases.randomize();

        Self {
            weights,
            biases,
            weights_grad,
            biases_grad,
            activation,
            output: None.into(),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Ref<Tensor> {
        // New scope for output.borrow_mut(). Idk if this is the best
        {
            let mut mut_borrow = self.output.borrow_mut();

            // If we already have output allocated and it's the right size
            if let Some(ref mut out) = *mut_borrow {
                if out.shape == vec![x.shape[0], self.weights.shape[1]] {
                    Tensor::matmul(out, &x, &self.weights);
                } else {
                    *mut_borrow = None;
                }
            }

            // Else we allocate new memory for the output
            if let None = *mut_borrow {
                *mut_borrow = Some(x.matmul_alloc(&self.weights));
            }

            // Add biases and activations
            if let Some(ref mut out) = *mut_borrow {
                out.add_self(&self.biases);
                self.activation(out);
            }
        }

        self.get_output_reference()
    }

    pub fn get_output_reference(&self) -> Ref<Tensor> {
        Ref::map(self.output.borrow(), |borrow| {
            borrow
                .as_ref()
                .expect("There is no output because the forward function was not called yet")
        })
    }

    pub fn backward(&mut self, layer_input: &Tensor, mut gradient: Tensor) -> Tensor {
        match self.activation {
            Activation::ReLU => gradient.relu(),
            Activation::Sigmoid => gradient.sigmoid_derivative(),
            Activation::SoftmaxCrossEntropy => (),
            Activation::LeakyReLU(c) => gradient.leaky_relu(c),
        }

        Tensor::matmul(
            &mut self.weights_grad,
            &layer_input.transpose_alloc(),
            &gradient,
        );

        self.weights_grad
            .elems
            .iter_mut()
            .for_each(|x| *x = *x / gradient.shape[0] as f32);

        for j in 0..gradient.shape[1] {
            let mut sum = 0.0;
            for i in 0..gradient.shape[0] {
                let index = i * gradient.shape[1] + j;
                sum += gradient.elems[index];
            }
            self.biases_grad.elems[j] = sum / gradient.shape[0] as f32;
        }

        gradient.matmul_alloc(&self.weights.transpose_alloc())
    }

    fn activation(&self, x: &mut Tensor) {
        match self.activation {
            Activation::ReLU => x.relu(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::SoftmaxCrossEntropy => x.softmax(),
            Activation::LeakyReLU(c) => x.leaky_relu(c),
        }
    }

    pub fn optimize(&mut self, lr: f32) {
        for i in 0..self.weights.elems.len() {
            self.weights.elems[i] -= lr * self.weights_grad.elems[i];
        }

        for i in 0..self.biases.elems.len() {
            self.biases.elems[i] -= lr * self.biases_grad.elems[i];
        }
    }
}
