"""
Quick test script to verify the inference pipeline
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def quick_test():
    """Run a quick test with a simple math problem"""

    print("Quick Test - LlaDA Model Inference\n")
    print("="*50)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    if device == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print("="*50)
    print("\nTest Problem:")
    problem = "What is 15 + 27?"
    print(problem)

    print("\nExpected Answer: 42")
    print("="*50)

    print("\nLoading model... (this may take a few minutes)")

    try:
        model_name = "GSAI-ML/LLaDA-8B-Base"

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        model.eval()

        print("Model loaded successfully!\n")

        # Create prompt
        prompt = f"""Solve the following math problem step by step.

Problem: {problem}

Solution: Let's solve this step by step.
"""

        print("Generating answer...")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()

        print("\n" + "="*50)
        print("Generated Solution:")
        print("="*50)
        print(answer)
        print("="*50)

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"\nError during test: {e}")
        print("\nPlease make sure:")
        print("1. You have installed all requirements: pip install -r requirements.txt")
        print("2. You have enough GPU memory (16GB+ recommended)")
        print("3. You have access to the model on HuggingFace")
        return False

    return True

if __name__ == "__main__":
    quick_test()
