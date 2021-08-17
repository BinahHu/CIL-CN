python -minclearn \
--data-path ../DER-ClassIL.pytorch/data/cifar100/ \
--options options/podnet/podnet_cnn_cifar100.yaml options/data/cifar100_1order.yaml\
--increment 10 \
--task-factor 1 \
--fixed-memory \
--device 0 \
--label podnet_cnn_cifar100_10steps_buffer1000 \
-resume /home/zhiyuan/CIL-backbone/incremental_learning.pytorch/results/dev/podnet/202107/week_5/20210730_podnet_cnn_cifar100_10steps/net_0_task_9.pth \
--logits-save-dir /home/zhiyuan/CIL-backbone/features/podnet_buffer2000_task