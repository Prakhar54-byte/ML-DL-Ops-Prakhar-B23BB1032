#include <iostream>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SDBenchmark");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Load the exported UNet ONNX model
    const char* model_path = "../sd-onnx/unet/model.onnx";
    
    std::cout << "Loading ONNX UNet model in C++..." << std::endl;
    Ort::Session session(env, model_path, session_options);

    // Dummy tensor setups for benchmarking shapes (Batch 2, Channels 4, 64x64)
    std::vector<int64_t> sample_shape = {2, 4, 64, 64};
    std::vector<float> sample_data(2 * 4 * 64 * 64, 0.5f);
    
    std::vector<int64_t> timestep_shape = {1};
    std::vector<int64_t> timestep_data = {10}; // Example timestep
    
    std::vector<int64_t> encoder_shape = {2, 77, 768};
    std::vector<float> encoder_data(2 * 77 * 768, 0.1f);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    
    Ort::Value sample_tensor = Ort::Value::CreateTensor<float>(memory_info, sample_data.data(), sample_data.size(), sample_shape.data(), sample_shape.size());
    Ort::Value timestep_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, timestep_data.data(), timestep_data.size(), timestep_shape.data(), timestep_shape.size());
    Ort::Value encoder_tensor = Ort::Value::CreateTensor<float>(memory_info, encoder_data.data(), encoder_data.size(), encoder_shape.data(), encoder_shape.size());

    const char* input_names[] = {"sample", "timestep", "encoder_hidden_states"};
    Ort::Value input_tensors[] = {std::move(sample_tensor), std::move(timestep_tensor), std::move(encoder_tensor)};
    const char* output_names[] = {"out_sample"};

    std::cout << "Starting 5 warm-up and inference iterations..." << std::endl;
    
    double total_time = 0.0;
    int runs = 5;

    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors, 3, output_names, 1);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        total_time += diff.count();
        std::cout << "Run " << i + 1 << " latency: " << diff.count() << " seconds\n";
    }

    std::cout << "\nAverage C++ ONNX UNet Inference Time: " << (total_time / runs) << " seconds\n";
    return 0;
}