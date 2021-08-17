python -minclearn \
--data-path ../DER-ClassIL.pytorch/data/cifar100/ \
--options options/podnet/podnet_cnn_cifar100.yaml options/data/cifar100_1order.yaml\
--increment 10 \
--task-factor 1 \
--fixed-memory \
--device 7 \
--label podnet_cnn_cifar100_10steps_buffer1000 \
-save last