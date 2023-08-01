use core::fmt;

use rand::{self, distributions::Standard, prelude::Distribution, Rng};

pub struct Tensor<T: num::Num> {
    shape: Vec<usize>,
    elems: Box<[T]>,
}

#[derive(Debug)]
pub enum TensorErrors {
    InvalidShape,
}

impl<T> Tensor<T>
where
    Standard: Distribution<T>,
    T: std::default::Default + std::clone::Clone + num::NumCast + num::Num + Copy,
{
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            shape,
            elems: vec![Default::default(); size].into_boxed_slice(),
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

    pub fn matmul(&self, other: &Tensor<T>) -> Result<Self, TensorErrors>
    where
        T: std::ops::AddAssign,
    {
        if self.shape.len() < 2 || other.shape.len() != 2 {
            return Err(TensorErrors::InvalidShape);
        }

        if self.shape.len() >= 3 {
            unimplemented!();
        }

        // Unwrap can't fail, because it has at least 2 elements
        if self.shape.last().unwrap() != &other.shape[0] {
            return Err(TensorErrors::InvalidShape);
        }

        let mut out_shape = self.shape.to_owned();

        if let Some(last) = out_shape.last_mut() {
            *last = other.shape[1];
        }

        let mut output = Tensor::new(out_shape);

        // TODO: Refactor so it works for multidimensional A matrix
        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                for k in 0..other.shape[0] {
                    let i1 = i * other.shape[1] + j;
                    let i2 = i * self.shape[1] + k;
                    let i3 = k * other.shape[1] + j;
                    //println!("{i} {j} {k}");
                    //println!("{i1} {i2} {i3}\n");
                    output.elems[i1] += self.elems[i2] * other.elems[i3];
                }
            }
        }

        Ok(output)
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
