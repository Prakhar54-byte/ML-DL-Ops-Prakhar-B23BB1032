import time
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_dir = "./sd-onnx"
print("Loading ONNX Model in Python...")
pipe = ORTStableDiffusionPipeline.from_pretrained(model_dir)

prompts = [
    "Bill Gates with a hoodie",
    "John Oliver with Naruto style",
    "Hello Kitty with Naruto style",
    "Lebron James with a hat",
    "A photograph of an orange cat with Naruto style"
]

latencies = []

for i, prompt in enumerate(prompts):
    start_time = time.time()
    image = pipe(prompt, num_inference_steps=20).images[0]
    end_time = time.time()
    
    latency = end_time - start_time
    latencies.append(latency)
    
    filename = f"out_py_{i}.png"
    image.save(filename)
    print(f"Generated {filename} in {latency:.2f} seconds.")

avg_latency = sum(latencies) / len(latencies)
print(f"\nAverage Python ONNX Inference Time: {avg_latency:.2f} seconds")