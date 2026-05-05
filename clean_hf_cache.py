
import os
import shutil
from pathlib import Path

# Common HuggingFace cache location on Windows
CACHE_DIR = Path(os.path.expanduser("~/.cache/huggingface/hub"))

# Models we want to remove
MODELS_TO_REMOVE = [
    "models--MBZUAI--LaMini-Flan-T5-248M",
    "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
]

def clean_cache():
    if not CACHE_DIR.exists():
        print(f"Cache directory not found at: {CACHE_DIR}")
        return

    print(f"Checking cache at: {CACHE_DIR}")
    
    freed_space = 0
    removed_count = 0
    
    for item in CACHE_DIR.iterdir():
        if item.is_dir() and item.name in MODELS_TO_REMOVE:
            print(f"Found unused model: {item.name}")
            try:
                # Calculate size
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                freed_space += size
                
                # Delete
                print(f"  Removing {item.name} ({size / (1024*1024):.2f} MB)...")
                shutil.rmtree(item)
                removed_count += 1
                print("  [OK] Removed.")
            except Exception as e:
                print(f"  [Error] Failed to remove {item.name}: {e}")

    if removed_count == 0:
        print("No unused models found. Cache is already clean!")
    else:
        print(f"\nSuccessfully removed {removed_count} models.")
        print(f"Reclaimed {freed_space / (1024*1024*1024):.2f} GB of disk space.")

if __name__ == "__main__":
    clean_cache()
