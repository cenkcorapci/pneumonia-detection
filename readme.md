# Pneumonia Detection with Deep Learning

A convolutional neural network model for [RSNA pneumonia detection challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) based on [Residual Networks](https://arxiv.org/abs/1512.03385).
 
### About the data
- You can check this [exploratory data analysis](https://github.com/cenkcorapci/pneumonia-detection/blob/master/eda.ipynb) 
notebook to see what the data set roughly looks like.
### How to use
- Download the [data set](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data).
- Set the paths in _config.py_
- Than either
   ```
   python encoder_decoder_model.py
   ```
    or use _resnet.ipynb_ which can also plot losses on each epoch on Jupyter notebook.

