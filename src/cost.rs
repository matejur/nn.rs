use crate::tensor::Tensor;

pub enum CostFunction {
    CrossEntropy,
    MeanSquaredError,
    SoftmaxCrossEntropy,
}

impl CostFunction {
    pub fn compute(&self, out: &Tensor, target: &Tensor) -> Tensor {
        match self {
            CostFunction::CrossEntropy => cross_entropy(out, target),
            CostFunction::SoftmaxCrossEntropy => todo!(),
            CostFunction::MeanSquaredError => todo!(),
        }
    }
}

// fn mse(out: &Tensor, target: &Tensor) -> f32 {
//     let mut diff = out.sub_alloc(&target);
//     diff.square();
//     diff.sum() / out.shape[0] as f32
// }

fn cross_entropy(out: &Tensor, target: &Tensor) -> Tensor {
    let mut loss = Tensor::new(out.shape.to_owned());
    for i in 0..out.shape[0] {
        for j in 0..out.shape[1] {
            let index = i * out.shape[1] + j;
            let e1 = target.elems[index];
            let e2 = out.elems[index].clamp(1e-5, 1.0 - 1e-5);
            loss.elems[index] = -e1 * e2.ln();
        }
    }

    loss
}
