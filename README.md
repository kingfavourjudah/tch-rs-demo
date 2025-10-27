# tch-rs-demo

A demonstration project showcasing PyTorch integration with Rust using the tch-rs library.

## Overview

This project provides a comprehensive demonstration of machine learning capabilities in Rust using PyTorch's C++ API through the tch-rs library. It showcases:

- Fundamental tensor operations and linear algebra
- Training a linear regression model from scratch
- Building and training a neural network to solve the XOR problem
- Gradient-based optimization using the Adam optimizer
- Automatic differentiation for backpropagation

Perfect for learning how to integrate PyTorch with Rust for high-performance machine learning applications.

## Prerequisites

- Rust (latest stable version)
- macOS ARM64 (Apple Silicon)
- LibTorch 2.5.1 or compatible version

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd tch-rs-demo
```

### 2. Download LibTorch

The project requires LibTorch to be available. Download the macOS ARM64 CPU version:

```bash
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.5.1.zip -o /tmp/libtorch.zip
unzip /tmp/libtorch.zip -d .
```

This will create a `libtorch` directory in the project root.

### 3. Build the project

```bash
LIBTORCH="./libtorch" cargo build
```

## Running the Project

To run the demo, use the following command:

```bash
LIBTORCH="./libtorch" \
LIBTORCH_BYPASS_VERSION_CHECK=1 \
DYLD_LIBRARY_PATH="./libtorch/lib" \
cargo run
```

### Expected Output

The program runs four comprehensive demonstrations:

1. **Basic Tensor Operations**: Addition, multiplication, mean, and standard deviation
2. **Matrix Operations**: Matrix multiplication and transpose operations
3. **Linear Regression**: Training a model to learn y = 3x + 2 from synthetic data
4. **Neural Network**: Training a simple network to learn the XOR function

Sample output:
```
PyTorch Rust (tch-rs) Comprehensive Demo
==========================================

=== Basic Tensor Operations ===
Tensor A: [1, 2, 3]
Tensor B: [4, 5, 6]
A + B: [5, 7, 9]
Mean of A: 2.0

=== Linear Regression Demo ===
Training for 200 epochs...
Epoch 200: Loss = 4.796352, Weight = 1.0024, Bias = 1.5473
True values:    Weight = 3.0000, Bias = 2.0000
Learned values: Weight = 1.0024, Bias = 1.5473

=== Simple Neural Network Demo ===
Training a neural network to learn XOR function
Epoch 1000: Loss = 0.000411
Input: [0, 0] => Expected: 0, Predicted: 0.0177
Input: [0, 1] => Expected: 1, Predicted: 0.9802
Input: [1, 0] => Expected: 1, Predicted: 0.9778
Input: [1, 1] => Expected: 0, Predicted: 0.0210
```

## Environment Variables

- **LIBTORCH**: Path to the LibTorch installation directory
- **LIBTORCH_BYPASS_VERSION_CHECK**: Set to `1` to bypass version compatibility checks
- **DYLD_LIBRARY_PATH**: Path to the LibTorch dynamic libraries (required at runtime on macOS)

## Project Structure

```
tch-rs-demo/
├── Cargo.toml          # Project configuration and dependencies
├── src/
│   └── main.rs         # Main application code
├── libtorch/           # LibTorch C++ library (downloaded separately)
└── README.md           # This file
```

## Dependencies

- **tch**: 0.18.0 - Rust bindings for PyTorch

## Features

### Tensor Operations
- Tensor creation from slices and arrays
- Element-wise arithmetic operations (addition, multiplication)
- Statistical operations (mean, standard deviation)
- Matrix operations (matmul, transpose)
- Tensor reshaping and manipulation

### Machine Learning
- **Linear Regression**: Implementation with gradient descent optimization
  - Synthetic data generation
  - Model parameter learning
  - Adam optimizer
  - Training loop with loss tracking

- **Neural Networks**: Multi-layer perceptron implementation
  - Sequential model building
  - Activation functions (ReLU, Sigmoid)
  - XOR problem solving
  - Forward and backward propagation

### Advanced Features
- Automatic differentiation
- Gradient-based optimization (Adam)
- VarStore for parameter management
- CPU computation support

## Troubleshooting

### Build Issues

If you encounter build errors, ensure:

1. LibTorch is properly downloaded and extracted to the `libtorch` directory
2. The `LIBTORCH` environment variable points to the correct path
3. You're using a compatible version of LibTorch (2.5.1 recommended)

### Runtime Issues

If the program fails to run:

1. Verify `DYLD_LIBRARY_PATH` is set correctly
2. Check that the LibTorch dynamic libraries exist in `libtorch/lib/`
3. Use `LIBTORCH_BYPASS_VERSION_CHECK=1` if you encounter version mismatch warnings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Resources

- [tch-rs Documentation](https://github.com/LaurentMazare/tch-rs)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Rust Documentation](https://doc.rust-lang.org/)
