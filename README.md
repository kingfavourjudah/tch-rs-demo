# tch-rs-demo

A demonstration project showcasing PyTorch integration with Rust using the tch-rs library.

## Overview

This project demonstrates basic tensor operations using PyTorch's C++ API through Rust bindings. It creates a simple tensor, performs arithmetic operations, and prints the results.

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

```
 2
 3
 4
[ CPUIntType{3} ]
```

This output shows a tensor `[1, 2, 3]` with 1 added to each element, resulting in `[2, 3, 4]`.

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

- Basic tensor creation using `Tensor::from_slice()`
- Tensor arithmetic operations
- Tensor printing and visualization

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

MIT License (or specify your license)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Resources

- [tch-rs Documentation](https://github.com/LaurentMazare/tch-rs)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Rust Documentation](https://doc.rust-lang.org/)
