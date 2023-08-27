use std::cell::{Ref, RefCell};

use crate::tensor::Tensor;

pub struct Linear {
    pub weights: Tensor,
    pub biases: Tensor,
    pub weights_grad: Tensor,
    pub biases_grad: Tensor,

    pub activation_function: Activation,

    intermediates: RefCell<Option<Tensor>>,
    activations: RefCell<Option<Tensor>>,
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    LeakyReLU(f32),
    Sigmoid,
    SoftmaxCrossEntropy,
    None,
}

impl Linear {
    pub fn new(shape: [usize; 2], activation_function: Activation) -> Self {
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
            activation_function,
            intermediates: None.into(),
            activations: None.into(),
        }
    }

    pub fn set_weights(&mut self, elems: &[f32]) {
        self.weights.set_data(elems);
    }

    pub fn set_biases(&mut self, elems: &[f32]) {
        self.biases.set_data(elems);
    }

    pub fn forward(&self, x: &Tensor) -> Ref<Tensor> {
        // New scope for output.borrow_mut(). Idk if this is the best
        {
            let mut intermediates_borrow = self.intermediates.borrow_mut();
            if let Some(ref mut inter) = *intermediates_borrow {
                if inter.shape == vec![x.shape[0], self.weights.shape[1]] {
                    Tensor::matmul(inter, x, &self.weights);
                    inter.add_self(&self.biases);
                } else {
                    *intermediates_borrow = None;
                }
            }

            // Else we allocate new memory for the output
            if let None = *intermediates_borrow {
                let mut inter = x.matmul_alloc(&self.weights);
                inter.add_self(&self.biases);
                *intermediates_borrow = Some(inter);
            }

            let mut activations_borrow = self.activations.borrow_mut();
            if let Some(ref mut act) = *activations_borrow {
                if act.shape == intermediates_borrow.as_ref().unwrap().shape {
                    act.set_data(&intermediates_borrow.as_ref().unwrap().elems);
                    self.activate(act);
                } else {
                    *activations_borrow = None;
                }
            }

            if let None = *activations_borrow {
                let mut act = Tensor::from_array(
                    intermediates_borrow.as_ref().unwrap().shape.to_owned(),
                    &intermediates_borrow.as_ref().unwrap().elems,
                );
                self.activate(&mut act);
                *activations_borrow = Some(act);
            }
        }

        self.get_activations_reference()
    }

    pub fn get_activations_reference(&self) -> Ref<Tensor> {
        Ref::map(self.activations.borrow(), |borrow| {
            borrow
                .as_ref()
                .expect("There is no output because the forward function was not called yet")
        })
    }

    pub fn backward(&mut self, inputs: &Tensor, mut gradient: Tensor) -> Tensor {
        let mut binding = self.intermediates.borrow_mut();
        let intermediates = binding.as_mut().unwrap();

        match self.activation_function {
            Activation::ReLU => gradient
                .elems
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = if intermediates.elems[i] > 0.0 { *x } else { 0.0 }),
            Activation::LeakyReLU(_) => todo!(),
            Activation::Sigmoid => gradient.elementwise_multiply(&intermediates),
            Activation::SoftmaxCrossEntropy => (),
            Activation::None => todo!(),
        }

        Tensor::matmul(&mut self.weights_grad, &inputs.transpose_alloc(), &gradient);

        self.weights_grad
            .scalar_multiply(1.0 / gradient.shape[0] as f32);

        for bias_index in 0..self.biases.shape[0] {
            let mut sum = 0.0;
            for sample in 0..gradient.shape[0] {
                let index = sample * gradient.shape[1] + bias_index;
                sum += gradient.elems[index];
            }
            self.biases_grad.elems[bias_index] = sum / gradient.shape[0] as f32;
        }

        gradient.matmul_alloc(&self.weights.transpose_alloc())
    }

    fn activate(&self, x: &mut Tensor) {
        match self.activation_function {
            Activation::ReLU => x.relu(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::SoftmaxCrossEntropy => x.softmax(),
            Activation::LeakyReLU(c) => x.leaky_relu(c),
            Activation::None => (),
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
