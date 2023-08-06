

## Environment
Tested on Ubuntu 16.04, python3.7, mxnet 1.1.0

Numpy, cv2 and matplotlib are required.

## Build Patch-Level Model
### Code
ReitnaNet modified based oh https://github.com/jkznst/RetinaNet-mxnet

### Dataset
Ref to https://mxnet.apache.org/versions/1.1.0/faq/recordio.html?highlight=rec and [SSD](https://github.com/zhreshold/mxnet-ssd) to build rec data

### Train
Modify necessary value in RetinaNet/run_train_cvs.sh

Then run the following command
```
cd RetinaNet
bash run_train_cvs.sh
```

## Build WSI-Level Model
### Data pkl file strucure
The folllowing structure is expected to train WSL-level model.
```
dataset/
├── normal
│   ├── WSI001.pkl
│   ├── WSI002.pkl 
│   ├── ...
├── asc_us
│   ├── ...
├── lsil
│   ├── ...
├── hsil
│   ├── ...
├── agc
│   ├── ...
```
### Pickle file content

Each pickle file contains metadata of the corresponding WSI image and its patch-level inferencing results

Here is an example of what the .pkl should look like
{ 
 "result": {  
   "lsil": [
    [xmin1,ymin1,xmax1,ymax1,conf1],
    [xmin2,ymin2,xmax2,ymax2,conf2],
    ...
   ],
   "hsil": [...],
   "asc_us": [...],
   "asc_h": [...],
   "scc": [...],
   "agc": [...],
 }, 
 "mpp": 0.25, 
 "size": (30000, 30000),
 }

### Train
Modify necessary value in Random_forest/exp.yaml
Then run the following command
```
cd Random_forest
python traintest.py --yml_path exp.yaml
```