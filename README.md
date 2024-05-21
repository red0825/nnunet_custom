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
1. Download the [M&Ms dataset](https://www.ub.edu/mnms/).
2. Run this command:
```
python3 .../nnunetv2/dataset_conversion/Dataset114_MNMs.py -i MNMS_FOLDER
```

### Step 2: export paths
Run this command:
```
export nnUNet_raw_data_base=".../nnUNet_raw"
export nnUNet_preprocessed=".../nnUNet_preprocessed"
export nnUNet_results=".../nnUNet_trained_models"
```

### Step 3: plan and preprocess
Run this command:
```
python3 plan_and_preprocess_script -t task_id --verify_dataset_integrity
```

### Step 4: train the dataset
Run this command:
```
python3 train_script.py -d task_id -tr CUSTOM_TRAINER -f FOLD
```

### Step 5: Inference
Run this command:
```
python3 predict_script.py -i INPUT_FOLDER -o OUTPUT_FOLDER -d task_id -tr CUSTOM_TRAINER -f FOLD --save_probabilities
```
