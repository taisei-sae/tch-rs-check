fn main() {
    println!("{:?}", tch::Device::cuda_if_available());
}
