# export HF_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# python /root/highspeedstorage/model_distill/SenseFlow/exp_sd35/hf_download.py

python scripts_utils/prepare_parquet_with_images.py \
    --parquet_dir /root/highspeedstorage/model_distill/SenseFlow/dataset/LAION_Aesthetics_1024/data \
    --output_dir /root/highspeedstorage/model_distill/SenseFlow/dataset/LAION_Aesthetics_1024/training_data \
    --max_samples 100000