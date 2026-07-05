#!/bin/bash
# Download Wan2.2 GGUF models for video engine
set -e

BASE="https://huggingface.co"

download() {
    local repo="$1"
    local file="$2"
    local dest="${3:-$file}"
    
    if [ -f "$dest" ]; then
        echo "✓ Already exists: $dest"
        return
    fi
    
    echo "Downloading $dest..."
    wget -q --show-progress "$BASE/$repo/resolve/main/$file" -O "$dest"
    echo "✓ Downloaded: $dest"
}

echo "=== Wan2.2 Model Download ==="
echo ""

# T2V 1.3B (smallest, best for quick testing)
download "Wan-AI/Wan2.2-T2V-1.3B-GGUF" "wan2.2-t2v-1.3b-Q4_0.gguf"

# I2V 14B (best quality, needs GPU)
download "Wan-AI/Wan2.2-I2V-A14B-GGUF" "wan2.2-i2v-a14b-Q4_0.gguf"

echo ""
echo "Done! Models ready for video_engine"
echo ""
echo "Usage:"
echo "  ./video_engine -m wan2.2-t2v-1.3b-Q4_0.gguf -p 'cat walking' -f 16"
