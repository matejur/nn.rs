use crate::tensor::Tensor;

pub struct Linear {
    pub weights: Tensor,
    pub biases: Tensor,
    pub weights_grad: Tensor,
    pub biases_grad: Tensor,
}

impl Linear {
    pub fn new(shape: [usize; 2]) -> Self {
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
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.matmul_alloc(&self.weights);
        out.add_self(&self.biases);

        out
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
