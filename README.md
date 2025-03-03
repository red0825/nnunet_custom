# nnunet_custom
Instructions to create a training for nnUNet.

## Docker image for nnUNet customization
All required packages are installed except for nnunet.
```
red0825/nnunet-custom:exclude-nnunet
```

## Training
The detail official instruction of nnUNet training configuration and usage can be found in: [nnUNet](https://github.com/MIC-DKFZ/nnunet)

Below is an example using the M&Ms dataset.

### Prepare data set for training and export paths
1. Download the [M&Ms dataset](https://www.ub.edu/mnms/).
2. Run this command:
```
export nnUNet_raw_data_base=".../nnUNet_raw"
export nnUNet_preprocessed=".../nnUNet_preprocessed"
export nnUNet_results=".../nnUNet_trained_models"
```
### Instructions
- Loss Custom
- UPL-SFDA 
