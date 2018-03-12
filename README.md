## Abstract
CROHME datasets originally exhibit features designed for _Online-handwritting_ recognition task.  
Apart from drawn traces being encoded, inkml files also contain trace drawing time captured.
So we need to extract new feature map, namely matrices of pixel intensities.

The following scripts will get you started with _Offline math symbols recognition_ task.


## Setup
All code is compatible with Python **3.5.*** version.

1. Extract **_CROHME_full_v2.zip_** (found inside **_data_** directory) contents before running any of the above scripts.

2. Install specified dependencies with pip (Python Package Manager) using the following shell command:
```
pip install -U -r requirements.txt
```


## Scripts info
1. **_extract.py_**
   - Extracts trace groups from inkml files.
   - Converts extracted trace groups into images. Images are **square shaped** bitmaps with only black (value 0) and white (value 1) pixels. Black color denotes patterns (ROI).
   - Labels those images (according to inkml files).
   - Flattens images to one-dimensional vectors.
   - Converts labels to one-hot format.
   - Dumps training and testing sets separately into **_outputs_** folder.

   **Command line arguments**: -b [BOX_SIZE] -d [DATASET_VERSION] -c [CATEGORY] -t [THICKNESS]

   **Example usage**: `python extract.py -b 50 -d 2011 2012 2013 -c digits lowercase_letters operators -t 5`

   **Caution**: Script doesn't work properly for images bigger than 200x200 (For yet unknown reason).

2. **_balance.py_** script balances the overall distribution of classes.

   **Command line arguments**: -b [BOX_SIZE] -ub [UPPER_BOUND][Optional]
   
   **Example usage**: `python balance.py -b 50 -ub 6000`

3. **_visualize.py_** script will plot single figure depicting a random batch of **extracted** data.

   **Command line arguments**: -b [BOX_SIZE] -n [N_SAMPLES] -c [COLUMNS]

    **Example usage**: `python visualize.py -b 50 -n 40 -c 8`

    **Sample Plot**:
    ![crohme_extractor_plot](https://user-images.githubusercontent.com/22115481/30137213-9c619b0a-9362-11e7-839a-624f08e606f7.png)

3. **_extract_hog.py_** script will extract **HoG features**.  
This script accepts 1 command line argument, namely **hog_cell_size**.  
**hog_cell_size** corresponds to **pixels_per_cell** parameter of **skimage.feature.hog** function.  
We use **skimage.feature.hog** to extract HoG features.  
Example of script execution: `python extract_hog.py 5`  <-- pixels_per_cell=(5, 5)  
This script loads data previously dumped by **_extract.py_** and again dumps its outputs(train, test) separately.


4. **_extract_phog.py_** script will extract **PHoG features**.  
For PHoG features, HoG feature maps using different cell sizes are concatenated into a single feature vector.  
So this script takes arbitrary number of **hog_cell_size** values(HoG features have to be previously extracted with **_extract_hog.py_**)  
Example of script execution: `python extract_phog.py 5 10 20` <-- loads HoGs with respectively 5x5, 10x10, 20x20 cell sizes.


5. **_histograms_** folder contains histograms representing **distribution of labels** based on different label categories. These diagrams help you better understand extracted data.


## Distribution of classes
![all_labels_distribution](https://cloud.githubusercontent.com/assets/22115481/26694312/413fb646-4707-11e7-943c-b8ecebd0c986.png)
Labels were combined from **_train_** and **_test_** sets.
