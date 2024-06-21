# nnunet_loss_custom
Instructions to create a training for nnUNet.

## Docker image for nnUNet customization
All required packages are installed except for nnunet.
```
red0825/nnunet-custom:nnunet-custom:exclude-nnunet
```

## Training
The detail official instruction of nnUNet training configuration and usage can be found in: [nnUNet](https://github.com/MIC-DKFZ/nnunet)

Below is an example using the M&Ms dataset.

### Step 1: Prepare data set for training
1. Download the [M&Ms dataset](https://www.ub.edu/mnms/).
2. Run this command:
```
python3 .../nnunetv2/dataset_conversion/Dataset114_MNMs.py -i MNMS_FOLDER
```

### Step 2: Export paths

Run this command:
```
export nnUNet_raw_data_base=".../nnUNet_raw"
export nnUNet_preprocessed=".../nnUNet_preprocessed"
export nnUNet_results=".../nnUNet_trained_models"
```

### Step 3: Plan and preprocess
Run this command:
```
python3 plan_and_preprocess_script -t 114 --verify_dataset_integrity
```

### Step 4: Train the dataset
#### Set CUSTOM_TRAINER. Without the -tr option, the default nnU-Net learning method is adopted.
#### I created a couple of tr options...
* nnUNetTrainerExponentialLogarithmicCosV2
* nnUNetTrainerLogCoshDice1000epochs
* nnUNetTrainerFocalTversky1000epochs
* nnUNetTrainerLogCoshDice1000epochs

Run this command(2D U-Net):
```
python3 train_script.py -d 114 -c 2d -tr CUSTOM_TRAINER -f all
```
Run this command(3D U-Net):
```
python3 train_script.py -d 114 -c 3d_fullres -tr CUSTOM_TRAINER -f all
```

### Step 5: Inference
#### Specify the paths to the INPUT FOLDER and OUTPUT FOLDER.
Run this command(2D U-Net):
```
python3 predict_script.py -i INPUT_FOLDER -o OUTPUT_FOLDER -d 114 -c 2d -tr CUSTOM_TRAINER -f all --save_probabilities
```
Run this command(3D U-Net):
```
python3 predict_script.py -i INPUT_FOLDER -o OUTPUT_FOLDER -d 114 -c 3d_fullres -tr CUSTOM_TRAINER -f all --save_probabilities
```
