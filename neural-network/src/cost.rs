use crate::tensor::Tensor;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum CostFunction {
    CrossEntropy,
    MeanSquaredError,
}

impl CostFunction {
    pub fn compute(&self, out: &Tensor, target: &Tensor) -> f32 {
        match self {
            CostFunction::CrossEntropy => cross_entropy(out, target),
            CostFunction::MeanSquaredError => mse(out, target),
        }
    }
}

fn mse(out: &Tensor, target: &Tensor) -> f32 {
    let mut diff = out.sub_alloc(&target);
    diff.square();
    diff.sum() / (2.0 * out.shape[0] as f32)
}

fn cross_entropy(out: &Tensor, target: &Tensor) -> f32 {
    if out.shape != target.shape {
        panic!(
            "Loss calculation requires both tensors to be of same shape. Got: {:?} {:?}",
            out.shape, target.shape
        );
    }

    let mut loss = 0.0;
    for i in 0..out.shape[0] {
        for j in 0..out.shape[1] {
            let index = i * out.shape[1] + j;
            let e1 = target.elems[index];
            let e2 = out.elems[index].clamp(1e-5, 1.0 - 1e-5);
            loss += -e1 * e2.ln();
            loss += -(1.0 - e1) * (1.0 - e2).ln();
        }
    }

    loss / out.shape[0] as f32
}
