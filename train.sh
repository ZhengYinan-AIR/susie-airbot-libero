export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=False
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=d25955a87aecc6d5b8fbfba215e8076918096dab


python scripts/train.py --config configs/base.py:base