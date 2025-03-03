import csv
import os
import random
from pathlib import Path
import sys

import nibabel as nib

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.paths import nnUNet_raw

def make_out_dirs(dataset_id: int, task_name="MNMs_vendor"):
    dataset_name = f"Dataset{dataset_id:03d}_{task_name}"

    out_dir = Path(nnUNet_raw.replace('"', "")) / dataset_name
    out_train_dir = out_dir / "imagesTr"
    out_labels_dir = out_dir / "labelsTr"
    out_test_dir = out_dir / "imagesTs"
    out_test_labels_dir = out_dir / "labelsTs"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)
    os.makedirs(out_test_labels_dir, exist_ok=True)

    return out_dir, out_train_dir, out_labels_dir, out_test_dir, out_test_labels_dir

def read_csv(csv_file: str):
    patient_info = {}

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        patient_index = headers.index("External code")
        ed_index = headers.index("ED")
        es_index = headers.index("ES")
        vendor_index = headers.index("Vendor")

        for row in reader:
            patient_info[row[patient_index]] = {
                "ed": int(row[ed_index]),
                "es": int(row[es_index]),
                "vendor": row[vendor_index],
            }

    return patient_info


# ------------------------------------------------------------------------------
# Conversion to nnUNet format
# ------------------------------------------------------------------------------
def convert_mnms(src_data_folder: Path, csv_file_name: str, dataset_id: int, task_name: str, vendor: str):
    out_dir, out_train_dir, out_labels_dir, out_test_dir, out_test_labels_dir = make_out_dirs(dataset_id, task_name=task_name)
    patients_train = [f for f in (src_data_folder / "Training" / "Labeled").iterdir() if f.is_dir()]
    patients_test = [f for f in (src_data_folder / "Testing").iterdir() if f.is_dir()]

    patient_info = read_csv(str(src_data_folder / csv_file_name))

    save_cardiac_phases(patients_train, patient_info, out_train_dir, out_labels_dir, vendor)
    save_cardiac_phases(patients_test, patient_info, out_test_dir, out_test_labels_dir, vendor)

    # There are non-orthonormal direction cosines in the test and validation data.
    # Not sure if the data should be fixed, or we should skip the problematic data.
    # patients_val = [f for f in (src_data_folder / "Validation").iterdir() if f.is_dir()]
    # save_cardiac_phases(patients_val, patient_info, out_train_dir, out_labels_dir)

    generate_dataset_json(
        str(out_dir),
        channel_names={
            0: "cineMRI",
        },
        labels={"background": 0, "LVBP": 1, "LVM": 2, "RV": 3},
        file_ending=".nii.gz",
        num_training_cases=len(patients_train) * 2,  # 2 since we have ED and ES for each patient
    )


def save_cardiac_phases(patients: list[Path], patient_info: dict[str, dict[str, int]], out_dir: Path, labels_dir: Path = None, vendor: str = ""):
    for patient in patients:
        print(f"Processing patient: {patient.name}")

        image = nib.load(patient / f"{patient.name}_sa.nii.gz")
        ed_frame = patient_info[patient.name]["ed"]
        es_frame = patient_info[patient.name]["es"]
        vendor_name = patient_info[patient.name]["vendor"]
        
        if vendor_name != vendor:
            continue
        
        save_extracted_nifti_slice(image, ed_frame=ed_frame, es_frame=es_frame, out_dir=out_dir, patient=patient)

        if labels_dir:
            label = nib.load(patient / f"{patient.name}_sa_gt.nii.gz")
            save_extracted_nifti_slice(label, ed_frame=ed_frame, es_frame=es_frame, out_dir=labels_dir, patient=patient)


def save_extracted_nifti_slice(image, ed_frame: int, es_frame: int, out_dir: Path, patient: Path):
    # Save only extracted diastole and systole slices from the 4D H x W x D x time volume.
    image_ed = nib.Nifti1Image(image.dataobj[..., ed_frame], image.affine)
    image_es = nib.Nifti1Image(image.dataobj[..., es_frame], image.affine)

    # Labels do not have modality identifiers. Labels always end with 'gt'.
    suffix = ".nii.gz" if image.get_filename().endswith("_gt.nii.gz") else "_0000.nii.gz"

    nib.save(image_ed, str(out_dir / f"{patient.name}_ED{suffix}"))
    nib.save(image_es, str(out_dir / f"{patient.name}_ES{suffix}"))


if __name__ == "__main__":
    import argparse

    class RawTextArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    parser = argparse.ArgumentParser(add_help=False, formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="The downloaded MNMs dataset dir. Should contain a csv file, as well as Training, Validation and Testing "
        "folders.",
    )
    parser.add_argument(
        "-c",
        "--csv_file_name",
        type=str,
        default="211230_M&Ms_Dataset_information_diagnosis_opendataset.csv",
        help="The csv file containing the dataset information.",
    )

    args = parser.parse_args()
    args.input_folder = Path(args.input_folder)

    print("Converting...")
    
    convert_mnms(args.input_folder, args.csv_file_name, 115, "MNMsA", "A")
    convert_mnms(args.input_folder, args.csv_file_name, 116, "MNMsB", "B")
    convert_mnms(args.input_folder, args.csv_file_name, 117, "MNMsC", "C")
    convert_mnms(args.input_folder, args.csv_file_name, 118, "MNMsD", "D")

    print("Done!")