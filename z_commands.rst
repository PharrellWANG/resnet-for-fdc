command for training
====================

.. code-block:: bash

    $ bazel-bin/resnet/resnet_main --train_data_path='/Users/Pharrell_WANG/data/two_classes/32x32_homo_edge/train.csv' \
                                   --log_root='/Users/Pharrell_WANG/workspace/models/resnet/log' \
                                   --train_dir='/Users/Pharrell_WANG/workspace/models/resnet/log/train' \
                                   --dataset='fdc' \
                                   --num_gpus=1

command for eval
================

.. code-block:: bash

    $ bazel-bin/resnet/resnet_main --eval_data_path='/Users/Pharrell_WANG/data/two_classes/32x32_homo_edge/validation.csv' \
                                   --log_root='/Users/Pharrell_WANG/workspace/models/resnet/log' \
                                   --eval_dir='/Users/Pharrell_WANG/workspace/models/resnet/log/eval' \
                                   --mode=eval \
                                   --eval_batch_count=140 \
                                   --dataset='fdc' \
                                   --num_gpus=0

