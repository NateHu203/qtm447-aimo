#!/bin/bash
# Run at the top of every Colab session:
#   !bash scripts/colab_setup.sh

set -e

echo "==> Installing dependencies..."
pip install -q -r requirements-colab.txt

echo "==> Pulling latest code..."
git pull origin main

echo "==> Linking Drive directories..."
# Assumes Drive is mounted at /content/drive and project root is /content/aimo-project
DRIVE_DIR="/content/drive/MyDrive/AIMO"
mkdir -p "$DRIVE_DIR/checkpoints" "$DRIVE_DIR/data/processed" "$DRIVE_DIR/logs" "$DRIVE_DIR/hf_cache"

# Symlink checkpoints and processed data so src/ writes directly to Drive
ln -sfn "$DRIVE_DIR/checkpoints" ./checkpoints
ln -sfn "$DRIVE_DIR/data/processed" ./data/processed

echo "==> Done. Colab environment ready."
echo "    Checkpoints -> $DRIVE_DIR/checkpoints"
echo "    Processed data -> $DRIVE_DIR/data/processed"
