use rand::Rng;

#[derive(Debug)]
pub struct Tensor {
    shape: Vec<usize>,
    elems: Box<[f32]>,
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

    pub fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        self.elems.iter_mut().for_each(|x| *x = rng.gen());
    }

    pub fn incrementing(&mut self) {
        for i in 0..self.elems.len() {
            self.elems[i] = i as f32;
        }
    }

    pub fn matmul(out: &mut Self, a: &Self, b: &Self) {
        if out.shape[0] != a.shape[0]
            && out.shape.iter().last().unwrap() != b.shape.iter().last().unwrap()
        {
            todo!(
                "Output shape doesn't match input shapes correctly! \n out={:?} a={:?} b={:?}",
                out.shape,
                a.shape,
                b.shape
            );
        }

        out.elems.fill(0 as f32);

        let size_out_first = *out.shape.iter().next().unwrap();
        let size_out_last = *out.shape.iter().last().unwrap();
        let size_between = *b.shape.iter().next().unwrap();

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

    pub fn add(out: &mut Self, a: &Self, b: &Self) {
        if out.shape != a.shape && a.shape != b.shape {
            todo!("Broadcasting?");
        }

        for i in 0..out.elems.len() {
            out.elems[i] = a.elems[i] + b.elems[i];
        }
    }

    pub fn add_alloc(&self, other: &Self) -> Self {
        if self.shape != other.shape {
            todo!("Broadcasting?");
        }

        let out_shape = self.shape.to_owned();

        let mut output = Tensor::new(out_shape);

        Tensor::add(&mut output, self, other);

        output
    }

    pub fn add_self(&mut self, other: &Self) {
        if self.shape != other.shape {
            todo!("Broadcasting?");
        }

        for i in 0..self.elems.len() {
            self.elems[i] += other.elems[i];
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

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    #[should_panic]
    fn matmul_shape_check() {
        let m1 = Tensor::new(vec![2, 4]);
        let m2 = Tensor::new(vec![3, 4]);
        let _ = m1.matmul_alloc(&m2);
    }

    #[test]
    fn matmul() {
        let m1 = Tensor::from_array(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let m2 = Tensor::from_array(vec![4, 1], &[4.0, 3.0, 2.0, 1.0]);
        let r = Tensor::from_array(vec![1, 1], &[20.0]);

        assert_eq!(m1.matmul_alloc(&m2), r);

        let r = Tensor::from_array(
            vec![4, 4],
            &[4.0, 8.0, 12.0, 16.0, 3.0, 6.0, 9.0, 12.0, 2.0, 4.0, 6.0, 8.0, 1.0, 2.0, 3.0, 4.0],
        );

        assert_eq!(m2.matmul_alloc(&m1), r);

        let m1 = Tensor::from_array(vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m2 = Tensor::from_array(vec![2, 5], &[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let r = Tensor::from_array(
            vec![3, 5],
            &[20.0, 17.0, 14.0, 11.0, 8.0, 50.0, 43.0, 36.0, 29.0, 22.0, 80.0, 69.0, 58.0, 47.0, 36.0],
        );

        assert_eq!(m1.matmul_alloc(&m2), r);
    }

    #[test]
    fn add() {
        let mut m1 = Tensor::from_array(vec![2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let m2 = Tensor::from_array(vec![2, 4], &[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let r = Tensor::from_array(vec![2, 4], &[9.0; 8]);
        assert_eq!(m1.add_alloc(&m2), r);

        m1.add_self(&m2);
        assert_eq!(m1, r);
    }

    #[test]
    fn sub() {
        let mut m1 = Tensor::from_array(vec![2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let m2 = Tensor::from_array(vec![2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let r = Tensor::new(vec![2, 4]);
        assert_eq!(m1.sub_alloc(&m2), r);

        m1.sub_self(&m2);
        assert_eq!(m1, r);
    }
}
