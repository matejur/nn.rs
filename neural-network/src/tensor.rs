use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub elems: Box<[f32]>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>) -> Self {
        if shape.len() < 1 {
            panic!("Shape can't have a length of 0");
        }

        if shape.len() > 3 {
            panic!("Only 1D and 2D tensors are currently supported!");
        }

        let size: usize = shape.iter().product();
        Tensor {
            shape,
            elems: vec![0 as f32; size].into_boxed_slice(),
        }
    }

    pub fn from_array(shape: Vec<usize>, elems: &[f32]) -> Self {
        if shape.iter().product::<usize>() != elems.len() {
            panic!("Product of shapes must match the length of array");
        }

        Tensor {
            shape,
            elems: elems.into(),
        }
    }

    pub fn set_data(&mut self, elems: &[f32]) {
        if elems.len() != self.elems.len() {
            panic!("New elements must be the same length as the old elems!");
        }

        self.elems.copy_from_slice(elems);
    }

    pub fn randomize(&mut self) {
        let mut rng = ChaCha8Rng::from_entropy();
        //let mut rng = ChaCha8Rng::seed_from_u64(42);
        self.elems
            .iter_mut()
            .for_each(|x| *x = rng.gen::<f32>() * 2.0 - 1.0);
    }

    pub fn incrementing(&mut self) {
        for i in 0..self.elems.len() {
            self.elems[i] = i as f32;
        }
    }

    pub fn matmul(out: &mut Self, a: &Self, b: &Self) {
        if out.shape[0] != a.shape[0]
            || out.shape.iter().last().unwrap() != b.shape.iter().last().unwrap()
        {
            todo!(
                "Output shape doesn't match input shapes correctly! \n out={:?} a={:?} b={:?}",
                out.shape,
                a.shape,
                b.shape
            );
        }

        out.elems.fill(0 as f32);

        let size_out_first = *out.shape.first().unwrap();
        let size_out_last = *out.shape.last().unwrap();
        let size_between = *b.shape.first().unwrap();

        for i in 0..size_out_first {
            for k in 0..size_between {
                for j in 0..size_out_last {
                    let i1 = i * size_out_last + j;
                    let i2 = i * size_between + k;
                    let i3 = k * size_out_last + j;
                    //println!("{i} {j} {k}");
                    //println!("{i1} {i2} {i3}\n");
                    out.elems[i1] += a.elems[i2] * b.elems[i3];
                }
            }
        }
    }

    pub fn matmul_alloc(&self, other: &Self) -> Self {
        if self.shape.last().unwrap() != &other.shape[0] {
            panic!("Last shape of first tensor must match with the first shape of the second!");
        }

        let mut out_shape = self.shape.to_owned();

        if let Some(last) = out_shape.last_mut() {
            *last = other.shape[1];
        }

        let mut output = Tensor::new(out_shape);

        Tensor::matmul(&mut output, self, other);

        output
    }

    pub fn transpose_alloc(&self) -> Tensor {
        if self.shape.len() != 2 {
            panic!("Currently only 2D tensor transpose implemented");
        }

        let mut out = Tensor::new(vec![self.shape[1], self.shape[0]]);

        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                let i1 = i * self.shape[1] + j;
                let i2 = j * self.shape[0] + i;
                out.elems[i2] = self.elems[i1];
            }
        }

        out
    }

    pub fn square(&mut self) {
        self.elems.iter_mut().for_each(|x| *x = x.powi(2));
    }

    pub fn sigmoid(&mut self) {
        self.elems
            .iter_mut()
            .for_each(|x| *x = 1.0 / (1.0 + f32::exp(-*x)));
    }

    pub fn sigmoid_derivative(&mut self) {
        self.sigmoid();
        self.elems.iter_mut().for_each(|x| *x = *x * (1.0 - *x));
    }

    pub fn relu(&mut self) {
        self.elems.iter_mut().for_each(|x| *x = x.max(0.0));
    }

    pub fn leaky_relu(&mut self, c: f32) {
        self.elems.iter_mut().for_each(|x| *x = x.max(c * *x));
    }

    pub fn elementwise_multiply(&mut self, other: &Self) {
        if self.shape != other.shape {
            panic!(
                "Can only elementwise multiply tensors of same shape. Got {:?} and {:?}",
                self.shape, other.shape
            );
        }

        self.elems
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = *x * other.elems[i]);
    }

    pub fn argmax(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for sample in 0..self.shape[0] {
            let mut max = 0.0;
            let mut max_index = 0;

            for i in 0..self.shape[1] {
                let index = sample * self.shape[1] + i;

                if self.elems[index] > max {
                    max = self.elems[index];
                    max_index = i;
                }
            }

            out.push(max_index);
        }

        out
    }

    pub fn softmax(&mut self) {
        let max = self.elems.iter().copied().reduce(f32::max).unwrap();
        for sample_index in 0..self.shape[0] {
            let mut denom = 0.0;
            for attrib_index in 0..self.shape[1] {
                let i = sample_index * self.shape[1] + attrib_index;
                self.elems[i] -= max;
                denom += self.elems[i].exp()
            }

            for attrib_index in 0..self.shape[1] {
                let i = sample_index * self.shape[1] + attrib_index;
                self.elems[i] = self.elems[i].exp() / denom;
            }
        }
    }

    pub fn sum(&self) -> f32 {
        self.elems.iter().sum()
    }

    pub fn scalar_multiply(&mut self, s: f32) {
        self.elems.iter_mut().for_each(|x| *x = s * *x);
    }

    pub fn add(out: &mut Self, a: &Self, b: &Self) {
        if out.shape != a.shape && a.shape != b.shape {
            if a.shape[a.shape.len() - 1] != b.shape[0] || b.shape.len() > 1 {
                panic!(
                    "Adding tensors of shapes {:?} and {:?} not supported",
                    a.shape, b.shape
                );
            }
        }

        for i in 0..out.elems.len() {
            out.elems[i] = a.elems[i] + b.elems[i % b.elems.len()];
        }
    }

    pub fn add_alloc(&self, other: &Self) -> Self {
        if self.shape != other.shape {
            if self.shape[self.shape.len() - 1] != other.shape[0] || other.shape.len() > 1 {
                panic!(
                    "Adding tensors of shapes {:?} and {:?} not supported",
                    self.shape, other.shape
                );
            }
        }

        let out_shape = self.shape.to_owned();

        let mut output = Tensor::new(out_shape);

        Tensor::add(&mut output, self, other);

        output
    }

    pub fn add_self(&mut self, other: &Self) {
        if self.shape != other.shape {
            if self.shape[self.shape.len() - 1] != other.shape[0] || other.shape.len() > 1 {
                panic!(
                    "Adding tensors of shapes {:?} and {:?} not supported",
                    self.shape, other.shape
                );
            }
        }

        for i in 0..self.elems.len() {
            self.elems[i] += other.elems[i % other.elems.len()];
        }
    }

    //TODO: turn in a macro?
    pub fn sub(out: &mut Self, a: &Self, b: &Self) {
        if out.shape != a.shape && a.shape != b.shape {
            todo!("Broadcasting?");
        }

        for i in 0..out.elems.len() {
            out.elems[i] = a.elems[i] - b.elems[i];
        }
    }

    pub fn sub_alloc(&self, other: &Self) -> Self {
        if self.shape != other.shape {
            todo!("Broadcasting?");
        }

        let out_shape = self.shape.to_owned();

        let mut output = Tensor::new(out_shape);

        Tensor::sub(&mut output, self, other);

        output
    }

    pub fn sub_self(&mut self, other: &Self) {
        if self.shape != other.shape {
            todo!("Broadcasting?");
        }

        for i in 0..self.elems.len() {
            self.elems[i] -= other.elems[i];
        }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.elems == other.elems
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            elems: self.elems.clone(),
        }
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn print_subtensor(
            f: &mut std::fmt::Formatter<'_>,
            shape: &[usize],
            elems: &[f32],
            indent_size: usize,
        ) -> std::fmt::Result {
            match shape {
                &[_] => {
                    let indent = " ".repeat(indent_size * 2);
                    writeln!(f, "{indent}{:?},", elems)
                }
                shape => {
                    let indent = " ".repeat(indent_size * 2);
                    writeln!(f, "{indent}[")?;
                    let outer_shape = shape[0];
                    let remaining_shape = &shape[1..];
                    let remaining_size: usize = remaining_shape.iter().product();
                    for i in 0..outer_shape {
                        let elems_to_print =
                            &elems[i * remaining_size..i * remaining_size + remaining_size];
                        print_subtensor(f, remaining_shape, elems_to_print, indent_size + 1)?;
                    }
                    writeln!(f, "{indent}],")
                }
            }
        }

        print_subtensor(f, &self.shape, &self.elems, 0)
    }
}

pub fn to_float(x: &[i32]) -> Vec<f32> {
    x.iter().map(|x| *x as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::{to_float, Tensor};

    #[test]
    #[should_panic]
    fn matmul_shape_check() {
        let m1 = Tensor::new(vec![2, 4]);
        let m2 = Tensor::new(vec![3, 4]);
        let _ = m1.matmul_alloc(&m2);
    }

    #[test]
    fn matmul() {
        let m1 = Tensor::from_array(vec![1, 4], &to_float(&[1, 2, 3, 4]));
        let m2 = Tensor::from_array(vec![4, 1], &to_float(&[4, 3, 2, 1]));
        let r = Tensor::from_array(vec![1, 1], &to_float(&[20]));

        assert_eq!(m1.matmul_alloc(&m2), r);

        let r = Tensor::from_array(
            vec![4, 4],
            &to_float(&[4, 8, 12, 16, 3, 6, 9, 12, 2, 4, 6, 8, 1, 2, 3, 4]),
        );

        assert_eq!(m2.matmul_alloc(&m1), r);

        let m1 = Tensor::from_array(vec![3, 2], &to_float(&[1, 2, 3, 4, 5, 6]));
        let m2 = Tensor::from_array(vec![2, 5], &to_float(&[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]));
        let r = Tensor::from_array(
            vec![3, 5],
            &to_float(&[20, 17, 14, 11, 8, 50, 43, 36, 29, 22, 80, 69, 58, 47, 36]),
        );

        assert_eq!(m1.matmul_alloc(&m2), r);
    }

    #[test]
    fn add() {
        let mut m1 = Tensor::from_array(vec![2, 4], &to_float(&[1, 2, 3, 4, 5, 6, 7, 8]));
        let m2 = Tensor::from_array(vec![2, 4], &to_float(&[8, 7, 6, 5, 4, 3, 2, 1]));
        let r = Tensor::from_array(vec![2, 4], &to_float(&[9; 8]));
        assert_eq!(m1.add_alloc(&m2), r);

        m1.add_self(&m2);
        assert_eq!(m1, r);
    }

    #[test]
    fn sub() {
        let mut m1 = Tensor::from_array(vec![2, 4], &to_float(&[1, 2, 3, 4, 5, 6, 7, 8]));
        let m2 = Tensor::from_array(vec![2, 4], &to_float(&[1, 2, 3, 4, 5, 6, 7, 8]));
        let r = Tensor::new(vec![2, 4]);
        assert_eq!(m1.sub_alloc(&m2), r);

        m1.sub_self(&m2);
        assert_eq!(m1, r);
    }

    #[test]
    fn add_broadcast() {
        let mut m1 = Tensor::from_array(vec![2, 4], &to_float(&[1, 2, 3, 4, 5, 6, 7, 8]));
        let m2 = Tensor::from_array(vec![4], &to_float(&[1, 2, 3, 4]));
        let r = Tensor::from_array(vec![2, 4], &to_float(&[2, 4, 6, 8, 6, 8, 10, 12]));
        assert_eq!(m1.add_alloc(&m2), r);

        m1.add_self(&m2);
        assert_eq!(m1, r);
    }
}
