use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

fn basic_tensor_operations() {
    println!("=== Basic Tensor Operations ===");

    // Create tensors
    let a = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    let b = Tensor::from_slice(&[4.0, 5.0, 6.0]);

    println!("\nTensor A:");
    a.print();

    println!("\nTensor B:");
    b.print();

    // Addition
    let sum = &a + &b;
    println!("\nA + B:");
    sum.print();

    // Multiplication
    let product = &a * &b;
    println!("\nA * B (element-wise):");
    product.print();

    // Mean and standard deviation
    let mean = a.mean(tch::Kind::Float);
    let std = a.std(false);
    println!("\nMean of A: {:?}", f64::try_from(mean).unwrap());
    println!("Std of A: {:?}", f64::try_from(std).unwrap());
}

fn matrix_operations() {
    println!("\n\n=== Matrix Operations ===");

    // Create matrices
    let matrix_a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    let matrix_b = Tensor::from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).reshape(&[3, 2]);

    println!("\nMatrix A (2x3):");
    matrix_a.print();

    println!("\nMatrix B (3x2):");
    matrix_b.print();

    // Matrix multiplication
    let matmul_result = matrix_a.matmul(&matrix_b);
    println!("\nMatrix A @ B (2x2):");
    matmul_result.print();

    // Transpose
    let transpose = matrix_a.transpose(0, 1);
    println!("\nTranspose of A:");
    transpose.print();
}

fn linear_regression() {
    println!("\n\n=== Linear Regression Demo ===");

    // Generate synthetic data: y = 3x + 2 + noise
    let true_weight = 3.0;
    let true_bias = 2.0;
    let num_samples = 100;

    // Create training data
    let x_train = Tensor::randn(&[num_samples, 1], tch::kind::FLOAT_CPU);
    let noise = Tensor::randn(&[num_samples, 1], tch::kind::FLOAT_CPU) * 0.1;
    let y_train = &x_train * true_weight + true_bias + noise;

    println!("\nGenerating synthetic data: y = {}x + {} + noise", true_weight, true_bias);
    println!("Training samples: {}", num_samples);

    // Initialize model parameters
    let vs = nn::VarStore::new(Device::Cpu);
    let weight = vs.root().var("weight", &[1], nn::Init::Randn { mean: 0.0, stdev: 1.0 });
    let bias = vs.root().var("bias", &[1], nn::Init::Const(0.0));

    // Optimizer
    let mut opt = nn::Adam::default().build(&vs, 1e-2).unwrap();

    // Training loop
    let epochs = 200;
    println!("\nTraining for {} epochs...", epochs);

    for epoch in 1..=epochs {
        // Forward pass
        let predictions = x_train.shallow_clone().matmul(&weight.reshape(&[1, 1])) + &bias;

        // Compute loss (Mean Squared Error)
        let loss = (&predictions - &y_train).pow_tensor_scalar(2).mean(tch::Kind::Float);

        // Backward pass and optimization
        opt.backward_step(&loss);

        // Print progress every 50 epochs
        if epoch % 50 == 0 {
            let loss_value: f64 = loss.try_into().unwrap();
            let w_value: f64 = weight.get(0).try_into().unwrap();
            let b_value: f64 = bias.get(0).try_into().unwrap();
            println!(
                "Epoch {}: Loss = {:.6}, Weight = {:.4}, Bias = {:.4}",
                epoch, loss_value, w_value, b_value
            );
        }
    }

    // Final results
    let final_weight: f64 = weight.get(0).try_into().unwrap();
    let final_bias: f64 = bias.get(0).try_into().unwrap();

    println!("\n=== Training Complete ===");
    println!("True values:    Weight = {:.4}, Bias = {:.4}", true_weight, true_bias);
    println!("Learned values: Weight = {:.4}, Bias = {:.4}", final_weight, final_bias);
    println!("Error:          Weight = {:.4}, Bias = {:.4}",
             (final_weight - true_weight).abs(),
             (final_bias - true_bias).abs());
}

fn neural_network_demo() {
    println!("\n\n=== Simple Neural Network Demo ===");

    // Generate XOR-like problem data
    let x_data = Tensor::from_slice2(&[
        [0.0f32, 0.0f32],
        [0.0f32, 1.0f32],
        [1.0f32, 0.0f32],
        [1.0f32, 1.0f32],
    ]);

    let y_data = Tensor::from_slice2(&[
        [0.0f32],
        [1.0f32],
        [1.0f32],
        [0.0f32],
    ]);

    println!("\nTraining a neural network to learn XOR function");
    println!("Input patterns: [[0,0], [0,1], [1,0], [1,1]]");
    println!("Expected output: [0, 1, 1, 0]");

    // Create a simple neural network
    let vs = nn::VarStore::new(Device::Cpu);
    let net = nn::seq()
        .add(nn::linear(&vs.root() / "layer1", 2, 4, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root() / "layer2", 4, 1, Default::default()))
        .add_fn(|xs| xs.sigmoid());

    let mut opt = nn::Adam::default().build(&vs, 1e-2).unwrap();

    // Training
    let epochs = 1000;
    println!("\nTraining for {} epochs...", epochs);

    for epoch in 1..=epochs {
        let predictions = net.forward(&x_data);
        let loss = (&predictions - &y_data).pow_tensor_scalar(2).mean(tch::Kind::Float);

        opt.backward_step(&loss);

        if epoch % 200 == 0 {
            let loss_value: f64 = loss.try_into().unwrap();
            println!("Epoch {}: Loss = {:.6}", epoch, loss_value);
        }
    }

    // Test the network
    println!("\n=== Testing Neural Network ===");
    let final_predictions = net.forward(&x_data);

    for i in 0..4 {
        let input: Vec<f64> = (0..2)
            .map(|j| x_data.get(i).get(j).try_into().unwrap())
            .collect();
        let expected: f64 = y_data.get(i).get(0).try_into().unwrap();
        let predicted: f64 = final_predictions.get(i).get(0).try_into().unwrap();

        println!(
            "Input: [{:.0}, {:.0}] => Expected: {:.0}, Predicted: {:.4}",
            input[0], input[1], expected, predicted
        );
    }
}

fn main() {
    println!("PyTorch Rust (tch-rs) Comprehensive Demo");
    println!("==========================================\n");

    // Run all demonstrations
    basic_tensor_operations();
    matrix_operations();
    linear_regression();
    neural_network_demo();

    println!("\n\nAll demonstrations completed successfully!");
}
