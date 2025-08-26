fn main() {
    println!("tch-rs CUDA Debug Info");
    println!("LIBTORCH path: {:?}", std::env::var("LIBTORCH"));
    println!(
        "LIBTORCH_USE_PYTORCH: {:?}",
        std::env::var("LIBTORCH_USE_PYTORCH")
    );
    println!(
        "LIBTORCH_BYPASS_VERSION_CHECK: {:?}",
        std::env::var("LIBTORCH_BYPASS_VERSION_CHECK")
    );
    println!("LD_LIBRARY_PATH: {:?}", std::env::var("LD_LIBRARY_PATH"));
    println!(
        "CUDA_VISIBLE_DEVICES: {:?}",
        std::env::var("CUDA_VISIBLE_DEVICES")
    );

    println!("\n Basic CUDA Checks");
    println!(
        "Device::cuda_if_available(): {:?}",
        tch::Device::cuda_if_available()
    );
    println!("Cuda::is_available(): {:?}", tch::Cuda::is_available());

    let device_count = tch::Cuda::device_count();
    println!("Cuda::device_count(): {}", device_count);

    println!("\nCUDA Device Creation Test");
    match std::panic::catch_unwind(|| {
        let device = tch::Device::Cuda(0);
        println!("CUDA device created: {:?}", device);

        let tensor = tch::Tensor::ones([2, 2], (tch::Kind::Float, tch::Device::Cpu));
        println!("Created CPU tensor: {:?}", tensor);

        let cuda_tensor = tensor.to_device(device);
        println!("Tensor moved to CUDA: {:?}", cuda_tensor);
    }) {
        Ok(_) => println!("CUDA operations completed successfully"),
        Err(e) => println!("CUDA operations panicked: {:?}", e),
    }

    println!("\nAdditional CUDA Tests");
    if tch::Cuda::is_available() {
        println!("CUDA is reported as available");
    } else {
        println!("CUDA is NOT available - investigating...");

        match std::panic::catch_unwind(|| {
            let _device = tch::Device::Cuda(0);
            println!("CUDA device 0 created successfully");
        }) {
            Ok(_) => println!("CUDA device creation succeeded despite is_available() being false"),
            Err(e) => println!("CUDA device creation failed: {:?}", e),
        }
    }
}
