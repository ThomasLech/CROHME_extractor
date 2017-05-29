## Abstract
CROHME datasets originally exhibit features needed for _Online-handwritting_ recognition.  
Apart from traces drawn encoded, inkml files also contain trace drawing time captured.
So we leave only those features that are needed for our task.

Following scripts will get you started in _Offline math symbols recognition_ task.

## Scripts info
1. **_parse.py_** script will extract **square-shaped** bitmaps.  
You can specify bitmap size with `bitmap_size` command line flag(argument).  
Patterns drawn are then **centered** inside **square-shaped** bitmaps.  
Example of script execution: `python parse.py 50`  <-- extracts 50x50 bitmaps.  
This script combines samples extracted from all training sets and all test sets respectively and dumps into 2 separate files.  


2. **_extract_hog.py_** script will extract **HoG features**.  
This script accepts 1 command line argument, namely **hog_cell_size**.  
**hog_cell_size** corresponds to **pixels_per_cell** parameter of **skimage.feature.hog** function.  
We use **skimage.feature.hog** to extract HoG features.  
Example of script execution: `python extract_hog.py 16`  <-- pixels_per_cell=(5, 5)  
This script loads data previously dumped by **_parse.py_** and again dumps its outputs(train, test) separately.


3. **_extract_phog.py_** script will extract **PHoG features**.  
For PHoG features, HoG feature maps using different cell sizes are concatenated into a single feature vector.  
So this script takes arbitrary number of **hog_cell_size** values(HoG features have to be previously extracted with **_extract_hog.py_**)  
Example of script execution: `python extract_phog.py 5 10 20` <-- loads HoGs with respectively 5x5, 10x10, 20x20 cell sizes.



## Installation
Python version we use is **3.5.0**.

1. Extract **_CROHME_full_v2.zip_**(found inside **_data_** directory) contents before running any of the above scripts.

2. Install specified dependencies with pip (Python Package Manager) using the following shell command:
```
pip install -U -r requirements.txt
```



## Labels distribution
![labels_histogram](https://cloud.githubusercontent.com/assets/22115481/26559054/6e731ee8-44ad-11e7-922b-20bd79210f2a.png)
Labels were combined from train and test sets.
