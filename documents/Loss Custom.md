## Loss Custom
This guide explains how to customize the loss function in nnU-Net for training segmentation models. The process involves setting up the dataset, preprocessing, training with custom loss functions, and performing inference.
### Step 1: Move data to the `nnUNet_raw_data_base` location and change the directory structure.
Run this command:
```
python3 .../nnunetv2/dataset_conversion/Dataset114_MNMs.py -i MNMS_FOLDER
```

### Step 2: Plan and preprocess
Run this command:
```
python3 plan_and_preprocess_script.py -t 114 --verify_dataset_integrity
```

### Step 3: Train the dataset
#### Set `LOSS_CUSTOM_TRAINER`. Without the `-tr` option, the default nnU-Net learning method is adopted.
#### I created a couple of `LOSS_CUSTOM_TRAINER`...

##### Loss custom
* `nnUNetTrainerExponentialLogarithmicCosV2`
* `nnUNetTrainerLogCoshDice1000epochs`
* `nnUNetTrainerFocalTversky1000epochs`
* `nnUNetTrainerLogCoshDice1000epochs`

Run this command(2D U-Net):
```
python3 train_script.py -d 114 -c 2d -tr LOSS_CUSTOM_TRAINER -f all
```
Run this command(3D U-Net):
```
python3 train_script.py -d 114 -c 3d_fullres -tr LOSS_CUSTOM_TRAINER -f all
```

### Step 4: Inference
#### Specify the paths to the `INPUT FOLDER` and `OUTPUT FOLDER`.
Run this command(2D U-Net):
```
python3 predict_script.py -i INPUT_FOLDER -o OUTPUT_FOLDER -d 114 -c 2d -tr LOSS_CUSTOM_TRAINER -f all --save_probabilities
```
Run this command(3D U-Net):
```
python3 predict_script.py -i INPUT_FOLDER -o OUTPUT_FOLDER -d 114 -c 3d_fullres -tr LOSS_CUSTOM_TRAINER -f all --save_probabilities
```

## Acknowledgments
This project is inspired by or based on the following works:

- **Original Paper**: S. Jadon, *A survey of loss functions for semantic segmentation*, IEEE CIBCB, 2020.