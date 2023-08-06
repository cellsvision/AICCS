###### base model train
cp symbol/symbol_builder_bak_100.py ./symbol/symbol_builder.py
python3 train.py --train-path path/to/train/rec \
                 --val-path path/to/val/rec \
                 --network resnet18 --batch-size 48 \
                 --pretrained ./model/resnet-18 --epoch 0 \
                 --prefix ./pcv/retina_res18_v1/retina_res18_v1 --gpus 0,1,2,3,4,5,6,7 \
                 --end-epoch 6 --data-shape 1024 --label-width 350 \
                 --optimizer sgd_cosine --lr 0.005 --momentum 0.9 --wd 0.0001 --lr-steps '5' --num-class 10 \
                 --class-names hsil,asc_h,lsil,asc_us,scc,agc,tri,actino,cc,candida \
                 --min_neg_samples 10 --warmup-epoch 1 --lr-factor 0.01 --tensorboard true --num-example num_of_data
cp symbol/symbol_builder_bak_25.py ./symbol/symbol_builder.py
python3 train.py --train-path path/to/train/rec \
                 --val-path path/to/val/rec \
                 --network resnet18 --batch-size 48 \
                 --pretrained ./model/resnet-18 --epoch 11 \
                 --prefix ./pcv/retina_res18_v1/retina_res18_v1 --gpus 0,1,2,3,4,5,6,7 \
                 --end-epoch 11 --data-shape 1024 --label-width 350 \
                 --optimizer sgd_cosine --lr 0.005 --momentum 0.9 --wd 0.0001 --lr-steps '5' --num-class 10 \
                 --class-names hsil,asc_h,lsil,asc_us,scc,agc,tri,actino,cc,candida \
                 --min_neg_samples 10 --warmup-epoch 1 --lr-factor 0.01 --tensorboard true --num-example num_of_data --resume 6
###### fine tune
cp symbol/symbol_builder_bak_25.py ./symbol/symbol_builder.py
python3 train.py --train-path path/to/train/rec  \
                 --val-path path/to/val/rec \
                 --network resnet18 --batch-size 48 \
                 --pretrained ./pcv/retina_res18_v1/retina_res18_v1 \
                 --epoch 11 \
                 --prefix ./pcv/retina_res18_v1/retina_res18_v1_1 \
                 --gpus 0,1,2,3,4,5,6,7 \
                 --end-epoch 11 \
                 --data-shape 1024 \
                 --label-width 350 \
                 --optimizer sgd_cosine \
                 --lr 0.005 \
                 --momentum 0.9 --wd 0.0001 --lr-steps '5' --num-class 10 \
                 --class-names hsil,asc_h,lsil,asc_us,scc,agc,tri,actino,cc,candida \
                 --min_neg_samples 10 --warmup-epoch 1 --lr-factor 0.01 --tensorboard true --num-example num_of_data
cp symbol/symbol_builder_bak_10.py ./symbol/symbol_builder.py
python3 train.py --train-path path/to/train/rec \
                 --val-path path/to/val/rec \
                 --network resnet18 --batch-size 48 \
                 --pretrained ./pcv/retina_res18_v1/retina_res18_v1 \
                 --epoch 11 \
                 --prefix ./pcv/retina_res18_v1/retina_res18_v1_1 \
                 --gpus 0,1,2,3,4,5,6,7 \
                 --end-epoch 21 \
                 --data-shape 1024 \
                 --label-width 350 \
                 --optimizer sgd_cosine \
                 --lr 0.005 \
                 --momentum 0.9 \
                 --wd 0.0001 --lr-steps '5' --num-class 10 \
                 --class-names hsil,asc_h,lsil,asc_us,scc,agc,tri,actino,cc,candida \
                 --min_neg_samples 10 \
                 --warmup-epoch 1 --lr-factor 0.01 \
                 --tensorboard true \
                 --num-example num_of_data \
                 --resume 11