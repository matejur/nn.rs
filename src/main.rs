mod tensor;

use tensor::Tensor;

fn main() {
    let m1 = Tensor::from_array(vec![1, 4], &[1, 2, 3, 4]);
    let m2 = Tensor::from_array(vec![4, 1], &[4, 3, 2, 1]);

    println!("{m1}");
    println!("{m2}");
    println!("{}", m2.matmul_alloc(&m1));
}
