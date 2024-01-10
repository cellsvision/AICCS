

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
```
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
```
### Train
Modify necessary value in Random_forest/exp.yaml
Then run the following command
```
cd Random_forest
python traintest.py --yml_path exp.yaml
```

### DEMO
1. goto https://ai-eng.cellsvision.com:3443 (make sure you are using chrome or edge )
![alt text](https://github.com/cellsvision/AICCS/blob/main/webui/%E5%9B%BE%E7%89%871.png)
2. login with user name :demouser and password:ai@cervixDeep
![alt text](https://github.com/cellsvision/AICCS/blob/main/webui/%E5%9B%BE%E7%89%872.png)
3. click “Slice Management” to review or check Cervix Cytology Slices (Whole Slide Images) under each datasets.
![alt text](https://github.com/cellsvision/AICCS/blob/main/webui/%E5%9B%BE%E7%89%873.png)
4. datasets list show each dataset id/folder name/created time/collection/number of slices and action buttons.
![alt text](https://github.com/cellsvision/AICCS/blob/main/webui/%E5%9B%BE%E7%89%874.png)
Button “Scan” means you can add slice images on server path to this dataset.
Button “Upload”used to upload slice images to this dataset(usually,this will be slow because of network issue.)
Button “Open”will open this dataset and display slice image list in this dataset.the slice image list will display Slice id/Slice Name/Slice collection(means where this slice sample collect from, like cervix)/Ai(result of Ai Analyze)/Quality(slice image sample quality)/Doctor Result(Doctor Result after manual review or audited),and Action Button “Open”will open an whole slide image online viewer to review/audit or manual check this slide image.
Action Button Analysis will add this image to Ai Analyze Task list.

#### Roi List in slice image viewer:
![alt text](https://github.com/cellsvision/AICCS/blob/main/webui/%E5%9B%BE%E7%89%875.png)
Click each roi can zoomto the roi to 40X .
![alt text](https://github.com/cellsvision/AICCS/blob/main/webui/%E5%9B%BE%E7%89%876.png)
Click each roi category list tabs will change the roi list to category selected.
![alt text](https://github.com/cellsvision/AICCS/blob/main/webui/%E5%9B%BE%E7%89%877.png)
Click the “i”icon in the right sidebar,you can check the “Medical Case Information” of this slice image and the metadata (just simple as name/width/height) of this image.and the label image of this slice.The AI analysis suggestion also displayed between “Case Information”and Image Information. 
![alt text](https://github.com/cellsvision/AICCS/blob/main/webui/%E5%9B%BE%E7%89%878.png)
![alt text](https://github.com/cellsvision/AICCS/blob/main/webui/%E5%9B%BE%E7%89%879.png)
Click the “document”icon in the right sidebar,you can edit the report of this case.
![alt text](https://github.com/cellsvision/AICCS/blob/main/webui/%E5%9B%BE%E7%89%8710.png)
Select each needed options and write your diagnosis. Before this you need to check the AI roi list or add Roi by label tools,the click right mouse key to the roi you selected ,you can add this roi to your report.
![alt text](https://github.com/cellsvision/AICCS/blob/main/webui/%E5%9B%BE%E7%89%8711.png)
