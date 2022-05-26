
# CUDA_VISIBLE_DEVICES=2 \
# python train_teacher.py --model vgg16 \
#                         --dataset cifar100 \
#                         --datapath /data/DataSets/ImageNet2012DataSets/ \
#                         --save_freq 5 \
#                         --epochs 150 \
#                         --batch_size 128 \
#                         --num_workers 4 \
#                         --learning_rate 0.02 \
#                         --lr_decay_epochs 50,90,130 \
#                         --lr_decay_rate 0.1 \
#                         --weight_decay 0.0005 \
#                         --trial 1 \
#                         --gpu 0 \
#                         -o res_t  \

CUDA_VISIBLE_DEVICES=3 \
python train_student.py --path_t res_t/models/MobileNetV2_cifar10_lr_0.01_decay_0.001_trial_1/MobileNetV2_best.pth \
                        --dataset cifar10 \
                        --datapath /data/DataSets/imagenet/ \
                        --distill PCA \
                        --model_s MobileNetV2PCA \
                        --save_freq 5 \
                        --learning_rate 0.01 \
                        --lr_decay_epochs 120,240,290 \
                        --lr_decay_rate 0.1 \
                        --weight_decay 0.0005 \
                        --epochs 300 \
                        --batch_size 64 \
                        --num_workers 4 \
                        -r 1 \
                        -a 0 \
                        -b 1 \
                        --eigenVar 0.95 \
                        --pcalayer -1 \
                        --alllayer \
                        --crit_type 'mse' \
                        --loss_type 'raw' \
                        --trial 1 \
                        -o res_s/once/raw/mse/mobilenetv2/0.95 \
                        --gpu 0 
                        # --multiprocessing-distributed \
                        # --dist-url 'tcp://127.0.0.1:12346' \
                        # --dist-backend 'nccl' \
                        # --world-size 1 \
                        # --rank 0 \