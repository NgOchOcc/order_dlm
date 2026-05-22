"""
Quick test for LLaDA Diffusion Model
"""

import torch
from transformers import AutoTokenizer, AutoModel
import sys

def test_llada():
    """Test LLaDA model with a simple math problem"""

    print("=" * 60)
    print("LLaDA Diffusion Model - Quick Test")
    print("=" * 60)

    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print("\n" + "=" * 60)

    # Model name
    model_name = "GSAI-ML/LLaDA-8B-Base"
    print(f"Loading model: {model_name}")

    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Set left padding for batch generation
        if tokenizer.padding_side != 'left':
            tokenizer.padding_side = 'left'

        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            trust_remote_code=True
        ).to(device)
        model.eval()

        print("Model loaded successfully!\n")

        # Test problem
        problem = "What is 15 + 27?"
        print(f"Test Problem: {problem}")
        print(f"Expected Answer: 42\n")

        # Create prompt (simple for Base model)
        prompt = f"{problem}\n\nAnswer:"

        print("Tokenizing...")
        encoded = tokenizer(
            [prompt],
            add_special_tokens=False,
            padding=False,
            return_tensors="pt"
        )
        input_ids = encoded['input_ids'].to(device)

        print(f"Input length: {input_ids.shape[1]} tokens")
        print(f"Generating with diffusion sampling...")
        print("This may take a minute...\n")

        # Import generation function
        sys.path.insert(0, '/Users/luungoc/Project/order_dlm')
        from inference_llada_math500_diffusion import generate_llada

        # Generate
        output_ids = generate_llada(
            model,
            input_ids,
            attention_mask=None,
            steps=64,  # Fewer steps for quick test
            gen_length=128,
            block_length=32,
            temperature=0.,
            cfg_scale=0.,
            remasking='low_confidence',
            mask_id=126336
        )

        # Decode
        generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)

        print("=" * 60)
        print("Generated Answer:")
        print("=" * 60)
        print(generated_text)
        print("=" * 60)

        print("\n✓ Test completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure you have installed: pip install -r requirements.txt")
        print("2. Make sure you have enough GPU memory (16GB+ recommended)")
        print("3. Make sure you have access to GSAI-ML/LLaDA-8B-Base on HuggingFace")
        return False

    return True


if __name__ == "__main__":
    test_llada()
