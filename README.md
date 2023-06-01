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
  - Data
    - `transform`: image transformations for the MNIST dataset
    - `dataset`: MNIST training and testing datasets
    - `inspect_batch()`: view a sample of a batch of the training dataset
  - Train
    - `train()`: train for one epoch on the training set
    - `test()`: test using the testing set
    - `plot_results()`: plot the training and testing loss and accuracy for each epoch


# Sample images
![mnist_sample](https://github.com/woncoh1/era1a5/assets/12987758/a1713d31-14fb-4345-91a1-bd1e2875b7bf)

# Model summary
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

# Training results
![training_results](https://github.com/woncoh1/era1a5/assets/12987758/babac61a-3e4b-46ae-8fb7-b077acea1d31)
