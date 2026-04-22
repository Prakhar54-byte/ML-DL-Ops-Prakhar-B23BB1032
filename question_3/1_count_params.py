from diffusers import UNet2DConditionModel
from peft import LoraConfig, get_peft_model

# Load Base UNet
unet = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet")
base_params = sum(p.numel() for p in unet.parameters())
print(f"Base UNet Parameters: {base_params:,}")

# Apply LoRA
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
unet_lora = get_peft_model(unet, lora_config)

lora_params = sum(p.numel() for p in unet_lora.parameters() if p.requires_grad)
total_params = base_params + lora_params

print(f"Trainable LoRA Parameters: {lora_params:,}")
print(f"Combined Parameter Count: {total_params:,}")