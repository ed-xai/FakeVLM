import os
from platform import processor
import time
import json
import torch
import argparse
from PIL import Image
from eval_single_image import load_model

"""
To run the code use:

python run_test_eval.py \
    --data_json_path /path/to/FakeClue_Data/data_json/test.json \
    --image_base_path /path/to/FakeClue_Data/test \
    --model_path /path/to/FakeVLM_Model \
    --quantization none \
    --device auto \
    --output_path new_test.json

"""


def process_fakeclue_with_fakevlm(
    data_json_path,
    image_base_path,
    model_path,
    quantization="none",
    device="auto",
    output_path=None
):
    """
    Process all images in FakeClue_Data using FakeVLM and add responses to test.json.
    
    Args:
        data_json_path: Path to test.json file
        image_base_path: Base path to FakeClue_Data
        model_path: Path to FakeVLM model
        quantization: Quantization mode (none, cuda-4bit, cuda-8bit)
        device: Device to use (auto, cpu, cuda)
        output_path: Path to save updated test.json (default: overwrite original)
    """
    
    # Load model
    print(f"Loading FakeVLM model from {model_path}...")
    processor, model, device = load_model(model_path, quantization, device)
    
    # Load test.json
    print(f"Loading test.json from {data_json_path}...")
    with open(data_json_path, 'r') as f:
        test_data = json.load(f)
    
    # Create model identifier string
    model_name = f'fakevlm_{quantization}'

    # Ensure final directory
    norm_image_base_path = os.path.normpath(image_base_path)
    if os.path.basename(norm_image_base_path) != 'test':
        image_base_path = os.path.join(image_base_path, 'test')
        if not os.path.exists(image_base_path):
            raise FileNotFoundError(f"Image base path not found: {image_base_path}")
    
    # Process each entry
    processed_count = 0
    failed_count = 0
    
    for idx, entry in enumerate(test_data):
        try:
            # Construct image path
            image_path = os.path.join(image_base_path, entry["image"])
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                failed_count += 1
                continue
            
            print(f"\n[{idx+1}/{len(test_data)}] Processing: {entry['image']}")
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Extract prompt from conversations
            prompt_value = entry["conversations"][0]["value"]
            
            # Run inference
            inputs = processor(
                text=prompt_value,
                images=image,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            start_time = time.time()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=256)
            elapsed = time.time() - start_time

            generated_ids = output[0][inputs["input_ids"].shape[1]:]
            response = processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Add FakeVLM response to conversations
            entry["conversations"].append({
                "from": model_name,
                "value": response,
                # "model": model_name,
                # "quantization": quantization,
                "inference_time": f"{elapsed:.2f}s"
            })
            
            print(f"Response added ({elapsed:.2f}s)")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {entry.get('image', 'unknown')}: {str(e)}")
            failed_count += 1
            continue
        
        # one case checking
        # break

    # Save updated test.json
    if output_path is None:
        output_path = data_json_path
    
    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(test_data)}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    return test_data


def parse_eval_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Run FakeVLM evaluation on FakeClue_Data")
    parser.add_argument(
        "--data_json_path",
        required=True,
        type=str,
        help="Path to test.json file"
    )
    parser.add_argument(
        "--image_base_path",
        required=True,
        type=str,
        help="Base path to FakeClue_Data"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to FakeVLM model"
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "cuda-4bit", "cuda-8bit"],
        help="Quantization mode: none (default), cuda-4bit/cuda-8bit (GPU only)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on: auto (default), cpu, or cuda"
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="Path to save updated test.json (default: overwrite original)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_eval_args()
    
    process_fakeclue_with_fakevlm(
        data_json_path=args.data_json_path,
        image_base_path=args.image_base_path,
        model_path=args.model_path,
        quantization=args.quantization,
        device=args.device,
        output_path=args.output_path
    )