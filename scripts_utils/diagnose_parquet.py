#!/usr/bin/env python3
"""
Diagnose parquet file structure and content.
"""

import sys
from pathlib import Path
import pandas as pd

if len(sys.argv) < 2:
    print("Usage: python diagnose_parquet.py /path/to/dataset.parquet")
    sys.exit(1)

parquet_path = Path(sys.argv[1])

if not parquet_path.is_file():
    print(f"Error: {parquet_path} not found")
    sys.exit(1)

print(f"Loading {parquet_path.name}...")
df = pd.read_parquet(parquet_path)

print(f"\n📊 Dataset Info:")
print(f"  Shape: {df.shape}")
print(f"\n📋 Columns: {len(df.columns)}")
for col in df.columns:
    dtype = df[col].dtype
    null_count = df[col].isnull().sum()
    print(f"  - {col}: {dtype} (nulls: {null_count})")

print(f"\n🔍 First row sample:")
first_row = df.iloc[0]
for col in df.columns:
    val = first_row[col]
    if col == "image":
        if isinstance(val, bytes):
            print(f"  - {col}: bytes ({len(val)} bytes)")
        elif hasattr(val, 'shape'):
            print(f"  - {col}: array with shape {val.shape}, dtype {val.dtype}")
        else:
            print(f"  - {col}: {type(val).__name__}")
    else:
        val_str = str(val)[:100]
        print(f"  - {col}: {val_str}")

# Try to load first image
if "image" in df.columns:
    print(f"\n🖼️  Attempting to load first image...")
    try:
        from PIL import Image
        from io import BytesIO
        
        image_data = df.iloc[0]["image"]
        
        if isinstance(image_data, bytes):
            img = Image.open(BytesIO(image_data))
            print(f"  ✓ Successfully opened image: size={img.size}, mode={img.mode}")
        elif hasattr(image_data, 'shape'):
            img = Image.fromarray(image_data.astype('uint8'))
            print(f"  ✓ Converted array to image: size={img.size}, mode={img.mode}")
        else:
            print(f"  ✗ Unknown image type: {type(image_data)}")
    except Exception as e:
        print(f"  ✗ Error loading image: {e}")
else:
    print(f"\n⚠️  No 'image' column found!")
    print(f"  Available columns: {list(df.columns)}")


import pandas as pd
df = pd.read_parquet('/root/highspeedstorage/model_distill/SenseFlow/dataset/LAION_Aesthetics_1024/data/train-00000-of-02043.parquet')
print(df.iloc[0]['image'])