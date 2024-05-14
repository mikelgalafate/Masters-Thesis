import argparse
import ast
import glob
import os

import nibabel as nib
import numpy as np


def preprocess_ct(path: str, new_range: list[int]):
    """Normalize CT volumes

    Apply a min-max normalization on CT volumes assuming an original range [-1024, 3071]
    Args:
        path (str): Path to the dataset
        new_range (list): New range for the volume
    """

    min_val = -1024
    max_val = 3071
    new_min = new_range[0]
    new_max = new_range[1]

    for vol_path in glob.glob(os.path.join(path, 'CT*/')):
        volume_name = glob.glob(os.path.join(vol_path, '*.nii*'))[0]
        print(f'Processing {volume_name}')
        volume = nib.load(volume_name)
        data = volume.get_fdata().clip(min_val, max_val)
        data = ((data - min_val) / (max_val - min_val) * (new_max - new_min)) + new_min
        new_volume = nib.Nifti1Image(data, affine=volume.affine, header=volume.header)
        nib.save(new_volume, volume_name)


def preprocess_mri(path: str, new_range: list[int]):
    """Normalize MRI volumes

    Apply a min-max normalization on MRI volumes using the min and max value of the volume.
    Args:
        path (str): Path to the dataset
        new_range (list): New range for the volume
    """

    new_min = new_range[0]
    new_max = new_range[1]

    for vol_path in glob.glob(os.path.join(path, 'T*/')):
        volume_name = glob.glob(os.path.join(vol_path, '*.nii*'))[0]
        print(f'Processing {volume_name}')
        volume = nib.load(volume_name)
        data = volume.get_fdata()
        min_val = np.min(data)
        max_val = np.max(data)
        data = ((data - min_val) / (max_val - min_val) * (new_max - new_min)) + new_min
        new_volume = nib.Nifti1Image(data, affine=volume.affine, header=volume.header)
        nib.save(new_volume, volume_name)


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser()
    choices = [name.split('_')[-1] for name, obj in locals().items() if callable(obj) and 'preprocess_' in name]
    parser_.add_argument('path',
                         type=str,
                         help='Path to the directory containing all subjects to be preprocessed.')
    parser_.add_argument('-m', '--mode',
                         type=str, choices=choices,
                         required=True,
                         help=f'Mode to which the preprocess will be applied. Available: {choices}')
    parser_.add_argument('-r', '--range',
                         type=lambda string: sorted(ast.literal_eval(string)),
                         required=True,
                         help=f'Range of the output volume. Format must be [min, max].')
    args = parser_.parse_args()

    if len(args.range) != 2:
        raise ValueError('Give one min and one max value.')

    for subject in os.listdir(args.path):
        sub_path = os.path.join(args.path, subject)
        if os.path.isdir(sub_path):
            globals()[f'preprocess_{args.mode}'](sub_path, args.range)
            print(f'Processed subject {subject}')
