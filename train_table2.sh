CUDA_VISIBLE_DEVICES=2,3 python main.py --timestamp 00000000000000 \
 --dataset CIFAR --exp_name ERM --split split --cpcc 1 --cpcc_metric l2 --seeds 1 \
 --task sub