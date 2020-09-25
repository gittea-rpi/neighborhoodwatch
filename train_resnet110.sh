#!/bin/bash

model=resnet110
python -u resnet_train.py  --arch=$model  --save-dir=save_$model

# Checking if things work on your system? Look no further
# model=resnet110
# python -u resnet_train.py --arch=$model --save-dir=save_$model --epochs=1 --save-every=1
# python test_resnet110.py
