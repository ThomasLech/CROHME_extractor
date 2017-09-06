## Abstract
CROHME datasets originally exhibit features designed for _Online-handwritting_ recognition task.  
Apart from drawn traces being encoded, inkml files also contain trace drawing time captured.
So we need to extract new feature map, namely matrices of pixel intensities.

The following scripts will get you started with _Offline math symbols recognition_ task.


## Setup
Python version we use is **3.5.0**.

1. Extract **_CROHME_full_v2.zip_**(found inside **_data_** directory) contents before running any of the above scripts.

2. Install specified dependencies with pip (Python Package Manager) using the following shell command:
```
pip install -U -r requirements.txt
```


## Scripts info
1. **_extract.py_** script will extract **square-shaped** bitmaps.  
With this script, you have more control over data being extracted, namely:
    * Extracting data belonging to certain dataset version.
    * Extracting certain categories of classes, like **digits** or **greek** (see categories.txt for details).
    
    **Usage**: `python extract.py <out_format> <box_size> <dataset_version=2013> <category=all>`

    **Example usage**: `python extract.py pixels 32 2011+2012+2013 digits+operators+lowercase_letters+greek`

    **Caution**: Other output formats than pixels, **do not** work yet.

2. **_visualize.py_** script will plot single figure containing a random batch of your **extracted** data.

    **Usage**: `visualize.py <number_of_samples> <number_of_columns=4>`

    **Example usage**: `python visualize.py 40 8`

    **Plot**:
    ![crohme_extractor_plot](https://user-images.githubusercontent.com/22115481/30137213-9c619b0a-9362-11e7-839a-624f08e606f7.png)

3. **_parse.py_** script will extract **square-shaped** bitmaps.  
You can specify bitmap size with `bitmap_size` command line flag(argument).  
Patterns drawn are then **centered** inside **square-shaped** bitmaps.  
Example of script execution: `python parse.py 50`  <-- extracts 50x50 bitmaps.  
This script combines samples extracted from all training sets and all test sets respectively and dumps into 2 separate files.  


4. **_extract_hog.py_** script will extract **HoG features**.  
This script accepts 1 command line argument, namely **hog_cell_size**.  
**hog_cell_size** corresponds to **pixels_per_cell** parameter of **skimage.feature.hog** function.  
We use **skimage.feature.hog** to extract HoG features.  
Example of script execution: `python extract_hog.py 5`  <-- pixels_per_cell=(5, 5)  
This script loads data previously dumped by **_parse.py_** and again dumps its outputs(train, test) separately.


5. **_extract_phog.py_** script will extract **PHoG features**.  
For PHoG features, HoG feature maps using different cell sizes are concatenated into a single feature vector.  
So this script takes arbitrary number of **hog_cell_size** values(HoG features have to be previously extracted with **_extract_hog.py_**)  
Example of script execution: `python extract_phog.py 5 10 20` <-- loads HoGs with respectively 5x5, 10x10, 20x20 cell sizes.


6. **_histograms_** folder contains histograms representing **distribution of labels** based on different label categories. These diagrams help you better understand extracted data.


## Distribution of labels
![all_labels_distribution](https://cloud.githubusercontent.com/assets/22115481/26694312/413fb646-4707-11e7-943c-b8ecebd0c986.png)
Labels were combined from **_train_** and **_test_** sets.
