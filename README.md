# TSAI ERA V1 A5: MNIST CNN
> Modularize training of a convolutional neural network on MNIST dataset

# Install
```
git clone https://github.com/woncoh1/era1a5.git
```

# Usage
- Run the notebook `S5.ipynb` to train and test the model
- Tweak the hyperparameters, prefixed with `params_`, for the dataloader and optimizer
- The convolutional neural network model is in the `model.py` module
- The MNIST dataset and the training engine are in the `utils.py` module

# Modules
- `model.py`
- `utils.py`
  - Dataset
  - Trainer

# Model
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
            Conv2d-2           [-1, 64, 24, 24]          18,432
         MaxPool2d-3           [-1, 64, 12, 12]               0
            Conv2d-4          [-1, 128, 10, 10]          73,728
            Conv2d-5            [-1, 256, 8, 8]         294,912
         MaxPool2d-6            [-1, 256, 4, 4]               0
            Linear-7                   [-1, 50]         204,800
            Linear-8                   [-1, 10]             500
================================================================
Total params: 592,660
Trainable params: 592,660
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.77
Params size (MB): 2.26
Estimated Total Size (MB): 3.03
----------------------------------------------------------------  
```
