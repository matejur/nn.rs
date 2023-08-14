use core::fmt;
use std::{
    fmt::Debug,
    ops::{AddAssign, SubAssign},
};

use num::{Num, NumCast};
use rand::{self, distributions::Standard, prelude::Distribution, Rng};

#[derive(Debug)]
pub struct Tensor<T: num::Num> {
    shape: Vec<usize>,
    elems: Box<[T]>,
}

impl<T> Tensor<T>
where
    Standard: Distribution<T>,
    T: Default + Num + NumCast + Copy + AddAssign + SubAssign + Debug,
{
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
            elems: vec![Default::default(); size].into_boxed_slice(),
        }
    }

    pub fn from_array(shape: Vec<usize>, elems: &[T]) -> Self {
        if shape.iter().product::<usize>() != elems.len() {
            eprintln!("Product of shapes must match the length of array");
            std::process::exit(1);
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
            self.elems[i] = num::NumCast::from(i).unwrap();
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

        out.elems.fill(num::NumCast::from(0).unwrap());

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

impl<T: Num> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.elems == other.elems
    }
}

impl<T: fmt::Debug + num::Num> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn print_subtensor<T: fmt::Debug>(
            f: &mut fmt::Formatter<'_>,
            shape: &[usize],
            elems: &[T],
            indent_size: usize,
        ) -> fmt::Result {
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
        let m1 = Tensor::<i32>::new(vec![2, 4]);
        let m2 = Tensor::<i32>::new(vec![3, 4]);
        let _ = m1.matmul_alloc(&m2);
    }

    #[test]
    fn matmul() {
        let m1 = Tensor::from_array(vec![1, 4], &[1, 2, 3, 4]);
        let m2 = Tensor::from_array(vec![4, 1], &[4, 3, 2, 1]);
        let r = Tensor::from_array(vec![1, 1], &[20]);

        assert_eq!(m1.matmul_alloc(&m2), r);

        let r = Tensor::from_array(
            vec![4, 4],
            &[4, 8, 12, 16, 3, 6, 9, 12, 2, 4, 6, 8, 1, 2, 3, 4],
        );

        assert_eq!(m2.matmul_alloc(&m1), r);

        let m1 = Tensor::from_array(vec![3, 2], &[1, 2, 3, 4, 5, 6]);
        let m2 = Tensor::from_array(vec![2, 5], &[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
        let r = Tensor::from_array(
            vec![3, 5],
            &[20, 17, 14, 11, 8, 50, 43, 36, 29, 22, 80, 69, 58, 47, 36],
        );

        assert_eq!(m1.matmul_alloc(&m2), r);
    }

    #[test]
    fn add() {
        let mut m1 = Tensor::from_array(vec![2, 4], &[1, 2, 3, 4, 5, 6, 7, 8]);
        let m2 = Tensor::from_array(vec![2, 4], &[8, 7, 6, 5, 4, 3, 2, 1]);
        let r = Tensor::from_array(vec![2, 4], &[9; 8]);
        assert_eq!(m1.add_alloc(&m2), r);

        m1.add_self(&m2);
        assert_eq!(m1, r);
    }

    #[test]
    fn sub() {
        let mut m1 = Tensor::from_array(vec![2, 4], &[1, 2, 3, 4, 5, 6, 7, 8]);
        let m2 = Tensor::from_array(vec![2, 4], &[1, 2, 3, 4, 5, 6, 7, 8]);
        let r = Tensor::new(vec![2, 4]);
        assert_eq!(m1.sub_alloc(&m2), r);

        m1.sub_self(&m2);
        assert_eq!(m1, r);
    }
}
