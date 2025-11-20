train_config_name=$1
model_name=$2
gpu_use=$3

export CUDA_VISIBLE_DEVICES=$gpu_use
echo $CUDA_VISIBLE_DEVICES
uv run torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py $train_config_name --exp-name=$model_name