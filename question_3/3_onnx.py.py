import torch
from diffusers import StableDiffusionPipeline
import os

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
lora_path = "./sd-naruto-lora"
merged_dir = "./sd-merged"
onnx_dir = "./sd-onnx"

print("Loading base model and fusing LoRA...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.load_lora_weights(lora_path)
pipe.fuse_lora()
pipe.save_pretrained(merged_dir)

print("Exporting to ONNX via Optimum...")
# Optimum CLI handles the complex graph tracing for SD
os.system(f"optimum-cli export onnx --model {merged_dir} --task stable-diffusion {onnx_dir}")

print("Export complete. Check the size of the original vs ONNX folders.")