python -minclearn \
--mode eval \
--options options/der/der_cnn_cifar100.yaml options/data/cifar100_1order_cluster5.yaml \
--increment 5 \
--device 4 \
--label eval \
-resume /home/zhiyuan/CIL-CN/results/dev/der/202110/week_2/20211013_der_buffer2000_pretrain/net_0_task_19.pth \
--logits-save-dir /home/zhiyuan/CIL-CN/features/der_cifar100_buffer2000_pretrained