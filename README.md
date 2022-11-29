# Project: Bounding Box

I am planning to write a model in TensorFlow that will be able to determine the bounding box for certain objects. In my early research I have found an airplane data set that I can use for my training.

# Team: Independent

# Research:
1. Caltech 101 dataset: https://data.caltech.edu/records/mzrjq-6wc02
a. I can use this dataset for my training images. If this doesn’t work there are many datasets on Kaggle specifically made for bounding box projects.
2. Original R-CNN paper: https://arxiv.org/abs/1311.2524
a. This is the original paper on R-CNN and should provide me with most of the information I need to understand the ideas behind generating bounding boxes for objects and how I want to think about architecting my model.
3. Basic example for bounding boxes: https://d2l.ai/chapter_computer-vision/bounding-box.htmla. This example provides a simple breakdown of how to do basic bounding boxes using a CNN, and I will obviously be able to expand upon it but it’s a good starting point.
4. Article with more advanced CNN bounding box regression walkthrough: 
https://medium.com/nerd-for-tech/building-an-object-detector-in-tensorflow-using-bounding-box-regression-2bc13992973fa. This article focuses on TensorFlow using a self-made dataset with red circles as the target shape. I think going more in depth with this article would help me understand how I should lay out my code when I actually start working on the project with a large data set.
