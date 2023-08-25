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
    Sigmoid,
    Softmax,
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
        {
            let mut mut_borrow = self.output.borrow_mut();
            if let Some(ref mut out) = *mut_borrow {
                if out.shape == vec![x.shape[0], self.weights.shape[1]] {
                    Tensor::matmul(out, &x, &self.weights);
                } else {
                    *mut_borrow = None;
                }
            }

            if let None = *mut_borrow {
                *mut_borrow = Some(x.matmul_alloc(&self.weights));
            }

            if let Some(ref mut out) = *mut_borrow {
                out.add_self(&self.biases);
                self.activation(out);
            }
        }

        Ref::map(self.output.borrow(), |borrow| borrow.as_ref().unwrap())
    }

    fn activation(&self, x: &mut Tensor) {
        match self.activation {
            Activation::ReLU => x.relu(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::Softmax => x.softmax(),
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
