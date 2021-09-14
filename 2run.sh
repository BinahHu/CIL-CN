python -minclearn \
--data-path ../CIL-backbone/DER-ClassIL.pytorch/data/cifar100/ \
--options options/joint/joint_cifar100_task.yaml options/data/cifar100_1order_cluster2.yaml \
--increment 10 \
--device 5 \
--workers 0 \
--label joint_cifar100_cluster2_10steps_task \
-save last \
-task