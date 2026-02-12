"""
Convert parquet dataset(s) with embedded images to SenseFlow-compatible format.

Expected parquet columns:
  - image: binary image data (PIL-readable format) or image array
  - prompt: text description
  - image_width: int (optional, for validation)
  - image_height: int (optional, for validation)
  - img_path: str (optional, existing path on disk)

Output: dataset.json for SenseFlow training

Usage:

  1) Extract from single parquet:
     python scripts_utils/prepare_parquet_with_images.py \
         --parquet_file /path/to/dataset.parquet \
         --output_dir ./training_data

  2) Extract from multiple parquets in a directory:
     python scripts_utils/prepare_parquet_with_images.py \
         --parquet_dir /path/to/parquet/directory \
         --output_dir ./training_data

  3) Use existing img_path (no extraction):
     python scripts_utils/prepare_parquet_with_images.py \
         --parquet_file /path/to/dataset.parquet \
         --output_dir ./training_data \
         --use_img_path

  4) Extract only N samples for testing:
     python scripts_utils/prepare_parquet_with_images.py \
         --parquet_file /path/to/dataset.parquet \
         --output_dir ./training_data \
         --max_samples 1000
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from io import BytesIO

import pandas as pd
from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow required. Install with: pip install Pillow")
    sys.exit(1)

try:
    import pyarrow.parquet as pq
except ImportError:
    print("Error: pyarrow required. Install with: pip install pyarrow")
    sys.exit(1)


def extract_images_from_parquet(
    parquet_path: str = None,
    parquet_dir: str = None,
    output_dir: Path = None,
    max_samples: int = 0,
    use_img_path: bool = False,
):
    """
    Extract images from parquet file(s) and create dataset JSON.
    
    Args:
        parquet_path: Path to single parquet file
        parquet_dir: Path to directory containing multiple parquet files
        output_dir: Output directory for images and JSON
        max_samples: Max samples to process (0 = all)
        use_img_path: If True, use img_path column instead of extracting images
    """
    parquet_files = []

    # Collect parquet files
    if parquet_path:
        p = Path(parquet_path)
        if not p.is_file():
            print(f"Error: {p} not found", file=sys.stderr)
            sys.exit(1)
        parquet_files = [p]
    elif parquet_dir:
        d = Path(parquet_dir)
        if not d.is_dir():
            print(f"Error: {d} not a directory", file=sys.stderr)
            sys.exit(1)
        parquet_files = sorted(d.glob("*.parquet")) + sorted(d.glob("**/*.parquet"))
        if not parquet_files:
            print(f"Error: No parquet files found in {d}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: Either --parquet_file or --parquet_dir required", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet file(s)")
    for f in parquet_files:
        print(f"  {f.name}")

    # Process files
    if use_img_path:
        return process_parquets_img_path(parquet_files, output_dir, max_samples)
    else:
        return process_parquets_extract(parquet_files, output_dir, max_samples)


def process_parquets_extract(parquet_files: list, output_dir: Path, max_samples: int):
    """Extract images from one or more parquets."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    keys = []
    image_paths = []
    prompts = []
    failed = 0
    global_idx = 0
    samples_processed = 0

    # Use batch processing to reduce memory usage
    BATCH_SIZE = 1000  # Process 1000 rows at a time

    for parquet_file in parquet_files:
        print(f"\nProcessing {parquet_file.name}...")
        
        # Use pyarrow to read parquet in batches for memory efficiency
        parquet_table = pq.ParquetFile(parquet_file)
        total_rows = parquet_table.metadata.num_rows
        print(f"  Total records: {total_rows:,}")
        
        # Validate columns (only read schema, not data)
        schema = parquet_table.schema_arrow
        column_names = [field.name for field in schema]
        
        if "prompt" not in column_names:
            print(f"  Warning: 'prompt' column not found, skipping", file=sys.stderr)
            continue
        if "image" not in column_names:
            print(f"  Warning: 'image' column not found, skipping", file=sys.stderr)
            continue
        
        # Process in batches
        for batch_idx, batch in enumerate(parquet_table.iter_batches(batch_size=BATCH_SIZE, columns=["image", "prompt"])):
            # Stop if max_samples reached
            if max_samples > 0 and samples_processed >= max_samples:
                break
            
            # Convert batch to pandas DataFrame
            df_batch = batch.to_pandas()

            pbar = tqdm(df_batch.iterrows(), total=len(df_batch), 
                       desc=f"  Batch {batch_idx+1}")
            for _, row in pbar:
                # Stop if max_samples reached
                if max_samples > 0 and samples_processed >= max_samples:
                    break

                prompt = row["prompt"]
                
                try:
                    image_data = row["image"]
                    
                    # Handle different image formats
                    if isinstance(image_data, dict):
                        # HuggingFace datasets format - dict with 'bytes' or 'path' key
                        if 'bytes' in image_data and image_data['bytes']:
                            img = Image.open(BytesIO(image_data['bytes']))
                        elif 'path' in image_data:
                            img = Image.open(image_data['path'])
                        else:
                            # Try any bytes-like key
                            bytes_data = None
                            for key in ['bytes', 'data', 'content', 'image']:
                                if key in image_data and image_data[key]:
                                    bytes_data = image_data[key]
                                    break
                            if bytes_data:
                                img = Image.open(BytesIO(bytes_data))
                            else:
                                failed += 1
                                continue
                    elif isinstance(image_data, bytes):
                        img = Image.open(BytesIO(image_data))
                    elif hasattr(image_data, 'tobytes') or hasattr(image_data, 'shape'):
                        # NumPy array or similar
                        if hasattr(image_data, 'shape') and len(image_data.shape) == 3:
                            img = Image.fromarray(image_data.astype('uint8'))
                        else:
                            failed += 1
                            continue
                    else:
                        # Try opening as PIL Image directly
                        if isinstance(image_data, (str, Path)):
                            img = Image.open(image_data)
                        else:
                            img = image_data

                    # Ensure RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Save image
                    fname = f"{global_idx:08d}.jpg"
                    img_path = images_dir / fname
                    img.save(img_path, quality=95)

                    # Record metadata
                    keys.append(f"{global_idx:08d}")
                    image_paths.append(str(img_path.resolve()))
                    prompts.append(str(prompt) if pd.notna(prompt) else "")

                    global_idx += 1
                    samples_processed += 1

                except Exception as e:
                    failed += 1
                    continue
            
            # Free memory after each batch
            del df_batch
            gc.collect()

        if max_samples > 0 and samples_processed >= max_samples:
            print(f"  Reached max_samples limit ({max_samples})")
            break

    if not keys:
        print("Error: No images extracted successfully", file=sys.stderr)
        sys.exit(1)

    print(f"\n✓ Extracted: {len(keys):,}  |  Failed: {failed:,}")

    # Generate JSON
    data = {
        "keys": keys,
        "image_paths": image_paths,
        "prompts": prompts,
    }

    json_path = output_dir / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved dataset.json ({len(keys):,} samples) → {json_path}")
    return json_path


def process_parquets_img_path(parquet_files: list, output_dir: Path, max_samples: int):
    """Create JSON using existing img_path column."""
    keys = []
    image_paths = []
    prompts = []
    missing = 0
    global_idx = 0
    samples_processed = 0

    for parquet_file in parquet_files:
        print(f"\nProcessing {parquet_file.name}...")
        df = pd.read_parquet(parquet_file)
        print(f"  Loaded {len(df):,} records")

        # Validate columns
        if "prompt" not in df.columns:
            print(f"  Warning: 'prompt' column not found, skipping", file=sys.stderr)
            continue
        if "img_path" not in df.columns:
            print(f"  Warning: 'img_path' column not found, skipping", file=sys.stderr)
            continue

        pbar = tqdm(df.iterrows(), total=len(df), desc=f"  Validating")
        for _, row in pbar:
            # Stop if max_samples reached
            if max_samples > 0 and samples_processed >= max_samples:
                break

            img_path = row["img_path"]
            prompt = row["prompt"]

            # Check if file exists
            if not Path(img_path).is_file():
                missing += 1
                continue

            keys.append(f"{global_idx:08d}")
            image_paths.append(str(Path(img_path).resolve()))
            prompts.append(str(prompt) if pd.notna(prompt) else "")

            global_idx += 1
            samples_processed += 1

        if max_samples > 0 and samples_processed >= max_samples:
            print(f"  Reached max_samples limit ({max_samples})")
            break

    if not keys:
        print("Error: No valid image paths found", file=sys.stderr)
        sys.exit(1)

    print(f"\n✓ Found: {len(keys):,}  |  Missing: {missing:,}")

    # Generate JSON
    data = {
        "keys": keys,
        "image_paths": image_paths,
        "prompts": prompts,
    }

    json_path = output_dir / "dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved dataset.json ({len(keys):,} samples) → {json_path}")
    return json_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet dataset(s) with images to SenseFlow format."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parquet_file", type=str,
                       help="Path to single parquet file")
    group.add_argument("--parquet_dir", type=str,
                       help="Path to directory containing multiple parquet files")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--use_img_path", action="store_true",
                        help="Use img_path column instead of extracting from image column")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to process (0 = all)")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = extract_images_from_parquet(
        parquet_path=args.parquet_file,
        parquet_dir=args.parquet_dir,
        output_dir=output_dir,
        max_samples=args.max_samples,
        use_img_path=args.use_img_path,
    )

    print(f"\n✓ Ready to use: {json_path}")


if __name__ == "__main__":
    main()
