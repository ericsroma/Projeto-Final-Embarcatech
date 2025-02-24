import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import pandas as pd
import json
import glob
import re
import shutil
import matplotlib.patches as patches

from scipy.ndimage import zoom
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from collections import Counter

from sklearn.model_selection import train_test_split


def butterworth(data: np.array, lowcut: int, highcut: int, fs: int, order: int):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def energy_reconstruction(IMFS_dict: dict, data: np.array, energy_threshold: float):
    Important_IMFS = {}
    relative_energies = {}
    preprocessed_signal = np.zeros(len(data))
    IMFS_sum = sum([IMFS_dict[key] for key in IMFS_dict.keys() if 'IMF' in key])
    res = IMFS_dict['res']
    for key, value in IMFS_dict.items():
        relative_energie = (np.sum(value**2)/(np.sum(IMFS_sum**2) + np.sum(res**2)))*100

        relative_energies[key] = relative_energie

        if relative_energie > energy_threshold:
            preprocessed_signal += value
            Important_IMFS[key] = relative_energie
    

    return Important_IMFS, preprocessed_signal, relative_energies


def get_mel_spectrogram(signal: np.array, sampling_rate: int, frame_length: float, frame_overlap: float, n_mels: int):
    frame_size = int(frame_length * sampling_rate)
    hop_length = int(frame_overlap * sampling_rate)
    mel_spectrogram = librosa.feature.melspectrogram(y = signal, sr = sampling_rate, n_fft = frame_size,  hop_length = hop_length, n_mels = n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    return mel_spectrogram, log_mel_spectrogram

def plot_spetrogram(spectrogram: np.array, sampling_rate: int, x_axis: str, y_axis: str):
    plt.figure(figsize = (25,10))
    librosa.display.specshow(spectrogram,
                             x_axis=x_axis,
                             y_axis=y_axis,
                             sr = sampling_rate,
                             cmap = 'viridis'
                            )
    #plt.colorbar(format = "%+2.f")
    plt.show()




def remove_intervals_between_beats(signal: np.array, min_segment_length:int, threshold: float):
    
    # removing intervals between beats
    almost_null_indices = np.where(np.abs(signal) < threshold)[0]

    segments = []
    start_idx = almost_null_indices[0]

    for i in range(1, len(almost_null_indices)):
        if almost_null_indices[i] != almost_null_indices[i-1] + 1:
            # End of a segment
            if i - np.where(almost_null_indices == start_idx)[0][0] >= min_segment_length:
                segments.append((start_idx, almost_null_indices[i-1]))
            start_idx = almost_null_indices[i]
    
    # Add the final segment if it's long enough
    if len(almost_null_indices) - np.where(almost_null_indices == start_idx)[0][0] >= min_segment_length:
        segments.append((start_idx, almost_null_indices[-1]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Signal', color='blue')
    height = np.max(signal)

    for segment in segments:
        start, end = segment
        plt.axvspan(start, end, color='red', alpha=0.3, label='Almost Null Segment' if segment == segments[0] else "")

    plt.show()

    return segments

def signal_windowing(signal: np.array, window_size: int):
    if len(signal) > 20000:
        signal = signal[:23000]

    if np.max(signal[:2000]) < 0.05:
        signal = signal[2000:] 
    signal_length = len(signal)
    if 3*window_size > signal_length:
        n_windows = (signal_length//window_size) + 1
        overlap = (signal_length - n_windows*window_size) // (n_windows-1)
    else:
        n_windows = (signal_length//window_size)
        overlap = (signal_length - n_windows*window_size) // (n_windows)
    res = signal_length%window_size

    start = 0
    windows = []
    start_end = []
    for i in range(n_windows):
        window = signal[start:start+window_size]
        start_end.append((start/8000,(start+window_size)/8000))
        if np.max(window) > 0.05:
            windows.append(window)
        start += (window_size + overlap)
    
    return windows, start_end

def get_mfcc_features(files_path: str, save_name:str, groups_df: pd.DataFrame, **kwargs)->pd.DataFrame:

    DBFolder = files_path

    Features = []
    labels = []
    groups = []
    files = []
    events = []
    file_names = []

    window_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k'}


    for filename in tqdm(os.listdir(DBFolder), desc = 'Processing_files'):
        filepath = os.path.join(DBFolder, filename)
        
        y, Fs = librosa.load(filepath, sr=8000) 

        filtered_signal = butterworth(data = y, lowcut = kwargs['filter_lowcut'], highcut = kwargs['filter_highcut'], fs = 8000, order =  2)

        windows = signal_windowing(signal = filtered_signal, window_size = kwargs['signal_window_size'])
        
        frame_length = int(kwargs['mel_frame_lenght'] * Fs)  
        hop_length = int(kwargs['mel_frame_overlap'] * Fs)    

        for i,window in enumerate(windows):
            c = librosa.feature.mfcc(y=window, sr=Fs, n_mfcc=kwargs['n_mfcc'], n_fft=frame_length, hop_length=hop_length, n_mels=kwargs['n_mels'])
            c = c.T  
            feature_vector = c.reshape(1, kwargs['n_mfcc']*c.shape[0])
            Features.append(feature_vector.flatten())
            labels.append(filename.split('_')[1])
            if groups_df is not None:
                groups.append(groups_df.at[filename, 'group'])
            files.append(int(filename.split('.')[0].split('_')[2]))
            events.append(window_map[i])
            file_names.append(filename)
        

    FeaturesX = pd.DataFrame(Features)
    labelsY = pd.DataFrame(labels, columns=['label'])
    groups = pd.DataFrame(groups, columns=['group'])
    files = pd.DataFrame(files, columns=['file'])
    events = pd.DataFrame(events, columns=['event'])
    file_names = pd.DataFrame(file_names, columns=['filename'])
    if groups_df is not None:
        Data = pd.concat([FeaturesX, labelsY, groups, files, events, file_names], axis=1)
    else:
        Data = pd.concat([FeaturesX, labelsY, files, events, file_names], axis=1)

    print(f"Final shape of the DataFrame: {Data.shape}")
    Data.to_csv(save_name, index=False)

    return Data

def get_spectrograms(Database_path: str, root_database_path: str, Spectrogram_database_path: str, **kwargs):
    train_path = os.path.join(Database_path, 'train')
    test_path = os.path.join(Database_path, 'test')
    collected_path = os.path.join(root_database_path, 'collected')
    physionet_path = os.path.join(root_database_path, 'PhysioNet')

    window_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k'}
    
    train_spectrograms = {}
    test_spectrograms = {}
    collected_spectrograms = {}
    physionet_spectrograms = {}

    sampling_rate = kwargs['sampling_rate']

    for filename in os.listdir(train_path):
        print(f"Processing: {filename} (train)")
        filepath = os.path.join(train_path, filename)
        
        y, Fs = librosa.load(filepath, sr=8000) 

        filtered_signal = butterworth(data = y, lowcut = kwargs['filter_lowcut'], highcut = kwargs['filter_highcut'], fs = kwargs['sampling_rate'], order =  kwargs['filter_order'])

        windows = signal_windowing(signal = filtered_signal, window_size = kwargs['signal_window_size'])
        
        for i,window in enumerate(windows):
            new_filename = f'{filename.split(".")[0]}_{window_map[i]}'
            mel_spectrogram, log_mel_spectrogram = get_mel_spectrogram(signal = window, sampling_rate = sampling_rate, frame_length=kwargs['mel_frame_lenght'], frame_overlap=kwargs['mel_frame_overlap'], n_mels = kwargs['n_mels'])
            train_spectrograms[new_filename] = log_mel_spectrogram

    
    for filename in os.listdir(test_path):
        print(f"Processing: {filename} (test)")
        filepath = os.path.join(test_path, filename)
        
        y, Fs = librosa.load(filepath, sr=8000) 

        filtered_signal = butterworth(data = y, lowcut = kwargs['filter_lowcut'], highcut = kwargs['filter_highcut'], fs = kwargs['sampling_rate'], order =  kwargs['filter_order'])

        windows = signal_windowing(signal = filtered_signal, window_size =  kwargs['signal_window_size'])
        
        for i,window in enumerate(windows):
            new_filename = f'{filename.split(".")[0]}_{window_map[i]}'
            mel_spectrogram, log_mel_spectrogram = get_mel_spectrogram(signal = window, sampling_rate = sampling_rate, frame_length=kwargs['mel_frame_lenght'], frame_overlap=kwargs['mel_frame_overlap'], n_mels = kwargs['n_mels'])         
            test_spectrograms[new_filename] = log_mel_spectrogram
    
    for filename in os.listdir(collected_path):
        print(f"Processing: {filename} (collected)")
        filepath = os.path.join(collected_path, filename)
        
        y, Fs = librosa.load(filepath, sr=8000) 

        filtered_signal = butterworth(data = y, lowcut = kwargs['filter_lowcut'], highcut = kwargs['filter_highcut'], fs = kwargs['sampling_rate'], order =  kwargs['filter_order'])

        windows = signal_windowing(signal = filtered_signal, window_size =  kwargs['signal_window_size'])
        
        for i,window in enumerate(windows):
            new_filename = f'{filename.split(".")[0]}_{window_map[i]}'
            mel_spectrogram, log_mel_spectrogram = get_mel_spectrogram(signal = window, sampling_rate = sampling_rate, frame_length=kwargs['mel_frame_lenght'], frame_overlap=kwargs['mel_frame_overlap'], n_mels = kwargs['n_mels'])         
            collected_spectrograms[new_filename] = log_mel_spectrogram

    for filename in os.listdir(physionet_path):
        print(f"Processing: {filename} (physionet)")
        filepath = os.path.join(physionet_path, filename)
        
        y, Fs = librosa.load(filepath, sr=8000) 

        filtered_signal = butterworth(data = y, lowcut = kwargs['filter_lowcut'], highcut = 100, fs = kwargs['sampling_rate'], order =  kwargs['filter_order'])

        windows = signal_windowing(signal = filtered_signal, window_size =  kwargs['signal_window_size'])
        
        for i,window in enumerate(windows):
            new_filename = f'{filename.split(".")[0]}_{window_map[i]}'
            mel_spectrogram, log_mel_spectrogram = get_mel_spectrogram(signal = window, sampling_rate = sampling_rate, frame_length=kwargs['mel_frame_lenght'], frame_overlap=kwargs['mel_frame_overlap'], n_mels = kwargs['n_mels'])         
            physionet_spectrograms[new_filename] = log_mel_spectrogram
    
    # save spectrograms
    output_npy_train = os.path.join(Spectrogram_database_path, 'npy', 'Train')
    os.makedirs(output_npy_train, exist_ok=True)
    output_png_train = os.path.join(Spectrogram_database_path, 'png', 'Train')
    os.makedirs(output_png_train, exist_ok=True)
    output_npy_test = os.path.join(Spectrogram_database_path, 'npy', 'Test')
    os.makedirs(output_npy_test, exist_ok=True)
    output_png_test = os.path.join(Spectrogram_database_path, 'png', 'Test')
    os.makedirs(output_png_test, exist_ok=True)
    output_npy_collected = os.path.join(Spectrogram_database_path, 'npy', 'Collected')
    os.makedirs(output_npy_collected, exist_ok=True)
    output_png_collected = os.path.join(Spectrogram_database_path, 'png', 'Collected')
    os.makedirs(output_png_collected, exist_ok=True)
    output_npy_physionet = os.path.join(Spectrogram_database_path, 'npy', 'physionet')
    os.makedirs(output_npy_physionet, exist_ok=True)
    output_png_physionet = os.path.join(Spectrogram_database_path, 'png', 'physionet')
    os.makedirs(output_png_physionet, exist_ok=True)
    
    
    for file, spectrogram in tqdm(train_spectrograms.items(), desc="Saving train files"):
        resampled_spectrogram = spectrogram_reshape(spectrogram = spectrogram, final_shape = kwargs['spectrogram_final_shape'])
        np.save(os.path.join(output_npy_train,f'{file}.npy'), resampled_spectrogram)
        plt.figure(figsize = (6,6))
        librosa.display.specshow(spectrogram, sr=sampling_rate, cmap='viridis')
        plt.axis('off')  
        plt.tight_layout()  
        plt.savefig(os.path.join(output_png_train,f'{file}.png'), bbox_inches='tight', pad_inches=0, dpi = 100)
        plt.close()

    for file, spectrogram in tqdm(test_spectrograms.items(), desc="Saving test files"):
        resampled_spectrogram = spectrogram_reshape(spectrogram = spectrogram, final_shape = kwargs['spectrogram_final_shape'])
        np.save(os.path.join(output_npy_test,f'{file}.npy'), resampled_spectrogram)
        plt.figure(figsize = (6,6))
        librosa.display.specshow(spectrogram, sr=sampling_rate, cmap='viridis')
        plt.axis('off')  
        plt.tight_layout()  
        plt.savefig(os.path.join(output_png_test,f'{file}.png'), bbox_inches='tight', pad_inches=0, dpi = 100)
        plt.close()

    for file, spectrogram in tqdm(collected_spectrograms.items(), desc="Saving collected files"):
        resampled_spectrogram = spectrogram_reshape(spectrogram = spectrogram, final_shape = kwargs['spectrogram_final_shape'])
        np.save(os.path.join(output_npy_collected,f'{file}.npy'), resampled_spectrogram)
        plt.figure(figsize = (6,6))
        librosa.display.specshow(spectrogram, sr=sampling_rate, cmap='viridis')
        plt.axis('off')  
        plt.tight_layout()  
        plt.savefig(os.path.join(output_png_collected,f'{file}.png'), bbox_inches='tight', pad_inches=0, dpi = 100)
        plt.close()

    for file, spectrogram in tqdm(physionet_spectrograms.items(), desc="Saving physionet files"):
        resampled_spectrogram = spectrogram_reshape(spectrogram = spectrogram, final_shape = kwargs['spectrogram_final_shape'])
        np.save(os.path.join(output_npy_physionet,f'{file}.npy'), resampled_spectrogram)
        plt.figure(figsize = (6,6))
        librosa.display.specshow(spectrogram, sr=sampling_rate, cmap='viridis')
        plt.axis('off')  
        plt.tight_layout()  
        plt.savefig(os.path.join(output_png_physionet,f'{file}.png'), bbox_inches='tight', pad_inches=0, dpi = 100)
        plt.close()



def save_train_test_split(Database_path, test_size):

    files = glob.glob(os.path.join(Database_path,'*', '*', '*.wav'))
    file_names = [os.path.basename(file) for file in files]
    file_labels = np.array([file.split('\\')[1] for file in files])
    file_groups = np.array([file.split('\\')[2] for file in files])
    df = pd.DataFrame({'name': file_names, 'label': file_labels, 'group': file_groups})

    # Get unique groups with their labels
    group_labels = df[['group', 'label']].drop_duplicates()

    # Initialize counters to store distributions
    label_counts = Counter(group_labels['label'])
    train_counts = {label: 0 for label in label_counts}
    test_counts = {label: 0 for label in label_counts}
    
    # Determine target sizes based on overall label distribution
    total_count = len(group_labels)
    train_target = {label: round((1 - test_size) * count) for label, count in label_counts.items()}
    test_target = {label: label_counts[label] - train_target[label] for label in label_counts}

    # Shuffle the data for randomness
    np.random.seed(42)
    shuffled_groups = group_labels.sample(frac=1).reset_index(drop=True)
    
    # Allocate groups to train or test
    train_groups, test_groups = set(), set()
    for _, row in shuffled_groups.iterrows():
        group, label = row['group'], row['label']
        if train_counts[label] < train_target[label]:
            train_groups.add(group)
            train_counts[label] += 1
        elif test_counts[label] < test_target[label]:
            test_groups.add(group)
            test_counts[label] += 1
    
    # Split the original dataframe based on the grouped train/test sets
    train_df = df[df['group'].isin(train_groups)]
    test_df = df[df['group'].isin(test_groups)]
    
    os.makedirs(os.path.join(Database_path, 'train'), exist_ok=True )
    os.makedirs(os.path.join(Database_path, 'test'), exist_ok=True)
    for file in files:
        group = file.split('\\')[2]
        if group in train_groups:
            shutil.copy(file, os.path.join(Database_path, 'train'))
        elif group in test_groups:
            shutil.copy(file, os.path.join(Database_path, 'test'))
    
    groups_df = df.set_index('name')
    
    return groups_df, train_df, test_df

def spectrogram_reshape(spectrogram: np.array, final_shape: tuple) -> np.array:
    zoom_factors = (final_shape[0] / spectrogram.shape[0], final_shape[1] / spectrogram.shape[1])

    resampled_spectrogram = zoom(spectrogram, zoom_factors, order=3) 

    return resampled_spectrogram



if __name__ == "__main__":
    Database_path = 'Database/Group_database'
    Spectrogram_database_path = 'Spectrogram_database'
    config = {'filter_lowcut': 25,          
            'filter_highcut': 600,
            'filter_order': 2,
            'n_mfcc': 19,
            'mel_frame_lenght': 0.01,
            'mel_frame_overlap': 0.001,
            'n_mels': 24,
            'energy_threshold': 3.5,
            'signal_window_size': 6000,
            'sampling_rate': 8000,
            'spectrogram_final_shape': (64,96)
    }

    get_mfcc_features(Database_path, n_mfcc = 19, n_mels = 24, f_length = 0.03, f_overlap= 0.01, save_name = 'CollectedData.csv')
    
    get_spectrograms(Database_path, Spectrogram_database_path, **config)

        