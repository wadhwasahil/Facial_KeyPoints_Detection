# Facial_KeyPoints_Detection

This repository contains code for the Kaggle problem.
[https://www.kaggle.com/c/facial-keypoints-detection](link)

**Training loss = 0.00819335**

**Score = 4.95813**


Files
* **conv.py** - This is a CNN architecture which is applied to each image in the dataset.
* **data_helpers.py** - This file loads the data(train/test) and also creates batches from it.
* **controller.py** - This files runs the main program i.e it loads the data from **data_helpers** and invokes the convolution architecture and finally stores the result in the csv file. This file performs both traning and testing.
