import os
import torch 
import argparse
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

def parse_args():
    parser = argparse.ArgumentParser(description="Single Image Inference for FakeVLM")
    parser.add_argument("--model_path", required=True, type=str, help="Path to the trained FakeVLM model")
    parser.add_argument("--image_path", required=True, type=str, help="Path to the image you want to test")
    parser.add_argument("--prompt", default="USER: <image>\nIs this image real or fake? Explain why. ASSISTANT:", type=str, help="Prompt for the model")
    return parser.parse_args()

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
    
    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        use_flash_attention_2=True, # Ensure flash-attn is installed
        revision='a272c74',
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)

    # Decode and print the output
    response = processor.decode(output[0], skip_special_tokens=True)
    
    # Clean up the prompt from the response if necessary
    if "ASSISTANT:" in response:
        response = response.split("ASSISTANT:")[-1].strip()
        
    print("\n--- MODEL OUTPUT ---")
    print(response)
    print("--------------------\n")
    
def main():
    args = parse_args()
    processor, model, device = load_model(args.model_path)
    infer_single_image(processor, model, device, args.image_path, args.prompt)

if __name__ == "__main__":
    main()