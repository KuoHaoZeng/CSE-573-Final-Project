#!/bin/bash
if [ ! -d tensorboard ]; then
   mkdir tensorboard
   virtualenv tensorboard
   source tensorboard/bin/activate
   pip install tensorboard tensorflow
   deactivate
fi
tensorboard/bin/tensorboard "$@"

