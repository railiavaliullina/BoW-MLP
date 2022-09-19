# BoW-MLP

## About The Project

1) Implementation of Bag Of Words and MLP,
2) Training MLP on AG News Classification Dataset for documents classification task.

## Getting Started


File to run:

    executor/executor.py

- After running executor.py validation will start on train and test data with the best checkpoint, and then training will continue.

To run on Kaggle: 

    https://www.kaggle.com/rvnrvn1/bow-mlp


## Additional Information

Visualization of accuracy on the training and test samples, loss are in: 

    saved_files/plots/accuracy_loss (all experiments)/

Confusion matrices are in: 

    saved_files/plots/conf_matrices/

Mlflow logs are in: 

    executor/mlruns.zip

Best achieved result:

    Accuracy: 90.5 %
    Testing error: 9.5 %

The best accuracy was obtained with 2 fc layers: 50000x128, 128x4 with dropouts, ReLU activation function, xavier weight initialization (experiment name in code and folders with visualization: 1_hidden_layer_128_dim).
