"""
GPU detection helper — stdlib only, safe to run before any pip install.
Outputs one of: 'cu124' | 'cpu'

CUDA 12.4 (cu124) is backward-compatible with RTX 20 / 30 / 40 / 50 series.
PyTorch 2.6+ is required for RTX 50 (Blackwell) kernel support.
"""
import subprocess
import sys


def detect_cuda_variant() -> str:
    """Return 'cu124' if an NVIDIA GPU is detected, 'cpu' otherwise."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cu124"
    except FileNotFoundError:
        pass  # nvidia-smi not on PATH
    except Exception:
        pass
    return "cpu"


if __name__ == "__main__":
    print(detect_cuda_variant())
