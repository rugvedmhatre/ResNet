# ResNet
ResNet Model trained to classify images on CIFAR-10 dataset

## Results

The model with the best accuracy is present in: [Notebooks/resnet-34-dropout.ipynb](https://github.com/rugvedmhatre/ResNet/blob/main/Notebooks/resnet-34-dropout.ipynb)

*Note: The resnet-34-dropout.ipynb file is also copied here in the base folder for easy evaluation*

| Parameter/Output | Value |
| ---------------- | ----- |
| Best Test Accuracy | **95.370%** |
| Total Trainable Parameters | **4,525,066** |
| Loss | CrossEntropyLoss |
| Optimizer | SGD |
| Learning Rate | 0.1 |
| Momentum | 0.9 |
| Weight Decay | 5e-4 |
| Training Epochs | 100 |
| Scheduler | CosineAnnealingLR |

The model is saved in this file: [Models/resnet_34_dropout_best.pth](https://github.com/rugvedmhatre/ResNet/blob/main/Models/resnet_34_dropout_best.pth)

This file stores: the model parameters with the best test accuracy, the value of the latest test accuracy, and the number of epochs at which we got the best test accuracy.

This file stores the output csv of the above mentioned model, which is uploaded on the Kaggle Leaderboard [Outputs/resnet_34_dropout_best_output.csv](https://github.com/rugvedmhatre/ResNet/blob/main/Outputs/resnet_34_dropout_best_output.csv)

## Testing the Model

First load the label names, and test file by making changes in the paths present in these variables in the notebook:

```python
# Specify the folder where the CIFAR-10 batch files are
cifar10_dir = './data/cifar-10-batches-py'
```

```python
# Load the label names
meta_data_dict = load_cifar_batch(os.path.join(cifar10_dir, 'batches.meta'))
```

```python
# Load test data
test_batch = load_cifar_batch('./cifar_test_nolabels.pkl')
```

Load the trained model by making changes in the path present in this variable in the notebook:

```python
# Load the trained model
model = ResNet34()

checkpoint = torch.load('./checkpoint/resnet_34_dropout_best.pth')
```

The output is saved at the following path:

```python
# Save the output in output.csv containing ID, Labels
output_data = {'ID': np.arange(len(predicted)), 'Labels': predicted.numpy()}
output_df = pd.DataFrame(output_data)
output_df.to_csv('resnet_34_dropout_best_output.csv', index=False)
```
