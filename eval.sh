python -minclearn \
--mode eval \
--data-path ../CIL-backbone/DER-ClassIL.pytorch/data/cifar100/ \
--options options/der/der_cnn_cifar100_task.yaml options/data/cifar100_1order_cluster.yaml \
--increment 10 \
--device 0 \
--label eval \
-resume /home/zhiyuan/CIL-CN/results/dev/der/202109/week_1/20210906_der_cifar100_cluster_10steps_task/net_0_task_9.pth \
--logits-save-dir /home/zhiyuan/CIL-backbone/features/der_task_cluster1 \
-task