python -minclearn \
--data-path ../CIL-backbone/DER-ClassIL.pytorch/data/cifar100/ \
--options options/der/der_cnn_cifar100_task.yaml options/data/cifar100_1order.yaml \
--increment 10 \
--device 6 \
--workers 0 \
--label der_cifar100_10steps_task \
-save last \
-task