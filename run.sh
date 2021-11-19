python -minclearn \
--options options/der/der_cnn_cifar100.yaml options/data/cifar100_1order_cluster5.yaml \
--increment 5 \
--device 1 \
--workers 16 \
--label der_buffer2000_cifar100_task_dogma \
-save last -savel -task