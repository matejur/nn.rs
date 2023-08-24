use crate::tensor::Tensor;

pub struct Linear {
    pub weights: Tensor,
    pub biases: Tensor,
    pub weights_grad: Tensor,
    pub biases_grad: Tensor,

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
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.matmul_alloc(&self.weights);
        out.add_self(&self.biases);

        self.activation(out)
    }

    fn activation(&self, mut x: Tensor) -> Tensor {
        match self.activation {
            Activation::ReLU => x.relu(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::Softmax => x.softmax(),
        }

        x
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
