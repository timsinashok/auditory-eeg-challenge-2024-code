import tensorflow as tf
import os
import numpy as np

def time_warping(eeg_data, alpha=0.02):
    """
    Apply time warping to EEG data.
    """
    num_channels = eeg_data.shape[1]
    num_samples = eeg_data.shape[0]
    
    # Generate random warping path
    path = np.cumsum(np.random.normal(0, alpha, num_samples))
    path = path - path.min()
    path = path / path.max() * (num_samples - 1)

    # Apply time warping to each channel
    augmented_data = np.zeros_like(eeg_data)
    for channel in range(num_channels):
        augmented_data[:, channel] = np.interp(np.arange(num_samples), path, eeg_data[:, channel])

    augmented_data = tf.constant(augmented_data, dtype=tf.float32)

    return augmented_data

def temporal_jittering(eeg_data, sigma=0.1):
    """
    Apply temporal jittering (random noise) to EEG data.
    """
    noise = np.random.normal(0, sigma, eeg_data.shape)
    augmented_data = eeg_data + noise

    augmented_data = tf.constant(augmented_data, dtype=tf.float32)

    return augmented_data


def augment_eeg_files(folder_path):
    # List all files in the input folder
    file_list = os.listdir(folder_path)

    # Iterate through each file in the folder
    for file_name in file_list:
        if file_name.endswith('.npy'):  # Assuming the files are in NumPy format
            # Load the original EEG data
            file_path = os.path.join(folder_path, file_name)
            eeg_data = np.load(file_path)

            # Augment the EEG data
            augmented_data_time_warping = time_warping(eeg_data)
            augmented_data_temporal_jittering = temporal_jittering(eeg_data)

            # Save augmented data to the same folder with modified filenames
            output_file_path_time_warping = os.path.join(folder_path, f'{file_name[:-4]}_time_warping.npy')
            np.save(output_file_path_time_warping, augmented_data_time_warping)

            output_file_path_temporal_jittering = os.path.join(folder_path, f'{file_name[:-4]}_temporal_jittering.npy')
            np.save(output_file_path_temporal_jittering, augmented_data_temporal_jittering)

if __name__ == "__main__":
    # Replace 'input_folder' with your actual folder path
    input_folder = '/scratch/at5282/split_data_aug'

    augment_eeg_files(input_folder)
