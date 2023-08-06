#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python3 $DIR/prepare_dataset.py --dataset pascal --year 2007,2012,2010 --set trainval --target /home/linzhen/diagnosis_data/lsil/voc_data/VOCpacked/train.lst --root /home/linzhen/diagnosis_data/lsil/voc_data/VOCdevkit --num-thread 8
python3 $DIR/prepare_dataset.py --dataset pascal --year 2007 --set test --target /home/linzhen/diagnosis_data/lsil/voc_data/VOCpacked/val.lst --shuffle False --root /home/linzhen/diagnosis_data/lsil/voc_data/VOCdevkit --num-thread 8
