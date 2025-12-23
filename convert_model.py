from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import torch

print("="*60)
print("MODEL CONVERSION SCRIPT")
print("="*60)

MODEL_PATH = "models/reddit_adventure_gpt_final"

print("\nStep 1: Checking if model directory exists...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR: Directory {MODEL_PATH} not found!")
    print("Please make sure you've extracted the model files correctly.")
    exit(1)
else:
    print(f"✓ Directory found: {MODEL_PATH}")

print("\nStep 2: Checking for model files...")
files_in_dir = os.listdir(MODEL_PATH)
print(f"Files found: {', '.join(files_in_dir)}")

has_safetensors = "model.safetensors" in files_in_dir
has_pytorch = "pytorch_model.bin" in files_in_dir

if not has_safetensors and not has_pytorch:
    print("\n❌ ERROR: No model weight files found!")
    print("Expected either 'model.safetensors' or 'pytorch_model.bin'")
    exit(1)

if has_pytorch:
    print("\n✓ pytorch_model.bin already exists!")
    print("No conversion needed.")
    exit(0)

print("\nStep 3: Loading model from safetensors...")
try:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH, use_safetensors=True)
    print("✓ Model loaded successfully from safetensors")
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    exit(1)

print("\nStep 4: Converting to PyTorch format...")
try:
    model.save_pretrained(
        MODEL_PATH,
        safe_serialization=False  # This creates pytorch_model.bin
    )
    print("✓ Conversion complete!")
except Exception as e:
    print(f"❌ ERROR during conversion: {e}")
    exit(1)

print("\nStep 5: Verifying converted files...")
print("\nAll files in model directory:")
for f in os.listdir(MODEL_PATH):
    size = os.path.getsize(os.path.join(MODEL_PATH, f))
    print(f"  {f}: {size / (1024*1024):.2f} MB")

if "pytorch_model.bin" in os.listdir(MODEL_PATH):
    print("\n" + "="*60)
    print("✓ SUCCESS! Model converted successfully!")
    print("You can now run: python backend.py")
    print("="*60)
else:
    print("\n❌ WARNING: pytorch_model.bin was not created")