use tch::Tensor;

fn main() {
    let tensor = Tensor::from_slice(&[1, 2, 3]) + 1;
    tensor.print(); // Should print [2, 3, 4]
}
