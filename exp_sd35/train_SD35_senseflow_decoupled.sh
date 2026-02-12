# export WANDB_API_KEY=wandb_api_key_here

work_path=$(dirname $0)
filename=$(basename $work_path)
T=$(date +%m%d%H%M)
OMP_NUM_THREADS=1 \
PYTHONFAULTHANDLER=True \
torchrun \
--nproc_per_node 4 \
--nnodes 1 \
main_trainer_sd35_senseflow_decoupled.py \
    /root/highspeedstorage/model_distill/SenseFlow/configs/SD35/sd35_senseflow.yaml \
    /root/highspeedstorage/model_distill/SenseFlow/exp_sd35/output/$T