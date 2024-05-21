# nnunet_loss_custom
Instructions to create a training for nnUNet.

## Docker image for nnUNet customization
All required packages are installed except for nnunet.
```
red0825/nnunet-custom:v3
```

## Training
The detail official instruction of nnUNet training configuration and usage can be found in: [nnUNet](https://github.com/MIC-DKFZ/nnunet)

### Step 1: prepare data set for training


### 2
Run this command:
```
export nnUNet_raw_data_base=".../nnUNet_raw_data_base"
export nnUNet_preprocessed=".../nnUNet_preprocessed"
export RESULTS_FOLDER=".../nnUNet_trained_models"
```

### 3
Run this command:
```
python3 plan_and_preprocess_script -t task_id --verify_dataset_integrity
```

### 4
Run this command:
```
python3 train_script.py -d task_id -tr CUSTOM_TRAINER -f FOLD
```

## 5
Run this command:
```
python3 predict_script.py -i INPUT_FOLDER -o OUTPUT_FOLDER -d task_id -tr CUSTOM_TRAINER -f FOLD --save_probabilities
```
