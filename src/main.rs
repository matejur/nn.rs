mod tensor;

use std::time::Instant;

use tensor::Tensor;

fn main() {
    let start = Instant::now();
    let mut m1: Tensor<f32> = Tensor::new(vec![128, 64]);
    m1.randomize();
    let duration = start.elapsed();

    println!("Creating first matrix: {:?}", duration);

    let start = Instant::now();
    let mut m2: Tensor<f32> = Tensor::new(vec![64, 128]);
    m2.randomize();
    let duration = start.elapsed();

    println!("Creating second matrix: {:?}", duration);

    let start = Instant::now();
    let _ = m1.matmul(&m2);
    let duration = start.elapsed();

    println!("Matmul: {:?}", duration);
}
