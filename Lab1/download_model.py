#!/usr/bin/env python3
"""Download Qwen3-ASR and Qwen3-TTS and Qwen3-4B models from HuggingFace"""

from huggingface_hub import snapshot_download

print("Downloading Qwen3-ASR-0.6B model...")
snapshot_download(
    repo_id="Qwen/Qwen3-ASR-0.6B",
    local_dir="./Qwen3-ASR-0.6B",
    local_dir_use_symlinks=False
)
print("Qwen3-ASR-0.6B Download complete!")

print("Downloading Qwen3-TTS-12Hz-0.6B-Base model...")
snapshot_download(
    repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    local_dir="./Qwen3-TTS-12Hz-0.6B-Base",
    local_dir_use_symlinks=False
)
print("Qwen3-TTS-12Hz-0.6B-Base Download complete!")

print("Downloading Qwen3-4B-quantized.w4a16 model...")
snapshot_download(
    repo_id="RedHatAI/Qwen3-4B-quantized.w4a16",
    local_dir="./Qwen3-4B-quantized.w4a16",
    local_dir_use_symlinks=False
)
print("Qwen3-4B-quantized.w4a16 Download complete!")