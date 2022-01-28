
# CUDA_VISIBLE_DEVICES=7 \
# python train_teacher.py --model ResNet34 \
#                         --dataset imagenet \
#                         --datapath /home/Datadisk/ImageNet2012DataSets \
#                         --save_freq 5 \
#                         --epochs 100 \
#                         --batch_size 256 \
#                         --num_workers 16 \
#                         --learning_rate 0.1 \
#                         --lr_decay_epochs 30,60,90 \
#                         -o temp \
#                         --resume res_t/models/ResNet34_imagenet_lr_0.1_decay_0.0005_trial_0/ResNet34_best.pth \
#                         --evaluate \
#                         --eigenVar 0.95 \
#                         --preact


CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train_student.py --path_t res_t/models/ResNet34_imagenet_lr_0.1_decay_0.0005_trial_0/ResNet34_best.pth \
                        --dataset imagenet \
                        --datapath /home/Datadisk/ImageNet2012DataSets/ \
                        --distill PCA \
                        --model_s ResNet34PCA \
                        --save_freq 5 \
                        --learning_rate 0.1 \
                        --lr_decay_epochs 30,60,90 \
                        --lr_decay_rate 0.1 \
                        --weight_decay 0.0005 \
                        --epochs 100 \
                        --batch_size 256 \
                        --num_workers 16 \
                        -r 1 \
                        -a 0 \
                        -b 0 \
                        --eigenVar 0.99 \
                        --pcalayer 4 \
                        --trial 0 \
                        -o res_s \
                        --multiprocessing-distributed \
                        --dist-url 'tcp://127.0.0.1:12346' \
                        --dist-backend 'nccl' \
                        --world-size 1 \
                        --rank 0 \