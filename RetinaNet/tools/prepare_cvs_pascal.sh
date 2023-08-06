#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python3 $DIR/prepare_dataset.py --dataset cvs --set train --target rec_data/train.lst --shuffle --root path/to/data --num-thread 8 --class-names hsil,asc_h,lsil,asc_us,scc,agc,tri,actino,cc,candida --true-negative
