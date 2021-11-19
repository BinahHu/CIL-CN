python -minclearn \
--options options/joint/joint_imagenet1000.yaml options/data/imagenet1000_1order.yaml \
--increment 1000 \
--device 0 1 \
--workers 64 \
--label joint_imagenet1000 \
--batch-size 256 \
-save last -savel