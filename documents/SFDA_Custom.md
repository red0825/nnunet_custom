## SFDA Custom
Source-Free Domain Adaptation (SFDA) is a technique that enables adaptation of a trained model to a new target domain **without access to the source domain data**. This is particularly useful in medical imaging, where privacy regulations or institutional constraints may restrict the sharing of source datasets. By using SFDA, we can fine-tune the model for domain shift while preserving data privacy.

This guide explains how to customize the Source-Free Domain Adaptation(SFDA) in nnU-Net for training segmentation models. The process involves setting up the dataset, preprocessing, training with custom loss functions, and performing inference.

In this process, **Dataset 115 serves as the source domain**, while **Datasets 116, 117, and 118 act as target domains**. Since SFDA does not have access to the source domain during adaptation, we use a pre-trained model from Dataset 115 and fine-tune it separately on each target domain (116, 117, and 118).

Each adaptation step involves:
1. **Preprocessing**: Ensuring consistency in dataset structures.
2. **Considering dataset plans**: Aligning preprocessing strategies between source and target datasets.
3. **Training**: Applying `nnUNetTrainerSFDA` for adaptation without accessing the source data.
4. **Inference**: Evaluating the adapted model on the target domain.

The following commands execute the SFDA adaptation for **each target dataset individually**:

### Step 1: Move data to the `nnUNet_raw_data_base` location and change the directory structure.
Run this command:
```
python3 .../nnunetv2/dataset_conversion/convert_MnMs.py -i MNMS_FOLDER
```

### Step 2: Plan and preprocess
Run this command:
```
python3 plan_and_preprocess_script.py -t 115 --verify_dataset_integrity
python3 plan_and_preprocess_script.py -t 116 --verify_dataset_integrity
python3 plan_and_preprocess_script.py -t 117 --verify_dataset_integrity
python3 plan_and_preprocess_script.py -t 118 --verify_dataset_integrity
```

### Step 3: Consider plans between datasets
When adapting a model to a new domain, differences in dataset characteristics (e.g., resolution, intensity distribution, anatomical structures) may impact performance. The script `consider_plans_between_datasets_script.py` ensures that the preprocessing and network architecture remain consistent across datasets. This step is crucial for **stabilizing training and improving generalization** to the new domain.

Since we are applying **SFDA to multiple target domains (116, 117, 118)**, we need to run the script separately for each target dataset.

Run these commands for each target dataset:

```
python3 consider_plans_between_datasets_script.py -s 115 -t 116 -sp nnUNetPlans -tp nnUNetPlans -ucn PlainConvUNetSFDA
```

### Step 3: Train the dataset
#### Set `SFDA_CUSTOM_TRAINER`. Without the `-tr` option, the default nnU-Net learning method is adopted.
#### I created a custom `SFDA_CUSTOM_TRAINER` for Source-Free Domain Adaptation.

##### SFDA custom
Currently, the following trainer is available:
* `nnUNetTrainerSFDA`

Run this command:

```
python3 train_script.py -d 116 -c 3d_fullres -tr SFDA_CUSTOM_TRAINER -f all
```

### Step 4: Inference
#### Specify the paths to the `INPUT FOLDER` and `OUTPUT FOLDER`.
Run this command:
```
python3 predict_sfda_script.py -i INPUT_FOLDER -o OUTPUT_FOLDER -d 116 -c 3d_fullres -tr SFDA_CUSTOM_TRAINER -f all --save_probabilities
```

## Acknowledgments
This project is inspired by or based on the following works:

- **Original Paper**: Wu, Jianghao, et al., *Upl-sfda: Uncertainty-aware pseudo label guided source-free domain adaptation for medical image segmentation.*, IEEE T-MI, 2023.
