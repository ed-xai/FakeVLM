import os
import time
import torch
import argparse
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Single Image Inference for FakeVLM")
    parser.add_argument("--model_path", required=True, type=str, help="Path to the trained FakeVLM model")
    parser.add_argument("--image_path", required=True, type=str, help="Path to the image you want to test")
    parser.add_argument("--prompt", default="USER: <image>\nIs this image real or fake? Explain why. ASSISTANT:", type=str, help="Prompt for the model")
    parser.add_argument("--quantization", default="none",
                        choices=["none", "cuda-4bit", "cuda-8bit"],
                        help="Quantization mode: none (default), cuda-4bit/cuda-8bit (GPU only)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to run on: auto (default), cpu, or cuda")
    return parser.parse_args()

def load_model(model_path, quantization="none", device_choice="auto"):
    print(f"Loading model from {model_path}...")

    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)

    # Resolve device
    use_cuda = torch.cuda.is_available()
    if device_choice == "auto":
        target_device = "cuda" if use_cuda else "cpu"
    else:
        target_device = device_choice

    if target_device == "cuda" and not use_cuda:
        raise RuntimeError("--device cuda requested, but no CUDA GPU was found.")
    if quantization != "none" and target_device != "cuda":
        raise RuntimeError(f"--quantization {quantization} requires --device cuda.")

    try:
        import flash_attn  # noqa: F401
        has_flash_attn = True
    except ImportError:
        has_flash_attn = False

    model_kwargs = dict(
        low_cpu_mem_usage=True,
        revision='a272c74',
    )

    if quantization == "cuda-4bit":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == "cuda-8bit":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif target_device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if target_device == "cuda" and has_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = LlavaForConditionalGeneration.from_pretrained(model_path, **model_kwargs)

    device = torch.device(target_device)
    if quantization == "none":
        model.to(device)

    model.eval()
    return processor, model, device

def infer_single_image(processor, model, device, image_path, prompt_text):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    # Process the inputs
    inputs = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt"
    )

    # Move inputs to the correct device (GPU/CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the output
    print("Generating response...")
    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    elapsed = time.time() - start

    # Decode and print the output
    response = processor.decode(output[0], skip_special_tokens=True)
    
    # Clean up the prompt from the response if necessary
    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:")[-1].strip()
        
    print("\n--- MODEL OUTPUT ---")
    print(response)
    print(f"--- {elapsed:.1f}s ---\n")
    
def main():
    args = parse_args()
    processor, model, device = load_model(args.model_path, args.quantization, args.device)
    infer_single_image(processor, model, device, args.image_path, args.prompt)

if __name__ == "__main__":
    main()