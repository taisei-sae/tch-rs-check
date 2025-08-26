fn main() {
    println!("Device: {:?}", tch::Device::cuda_if_available());
    println!("Is CUDA available: {:?}", tch::Cuda::is_available());
}
