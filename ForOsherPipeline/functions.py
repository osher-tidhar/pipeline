import pandas as pd
# import fireducks.pandas as pd

import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import re
from datetime import timedelta, datetime
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from nolds import sampen
from sklearn.cluster import KMeans
import pickle
import os
import neurokit2 as nk
import ast
# import pyhrv
from tqdm import tqdm
import copy
from scipy.stats import trim_mean, skew, kurtosis
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from collections import deque
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold
import concurrent.futures
from scipy.cluster.hierarchy import linkage
import seaborn as sns
from dateutil import rrule
from datetime import datetime
import matplotlib.patches as mpatches
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor



def RWL(signal, diff_flag=1):
        # RWL are the locations of the R-Peaks, I converted them into a
        # single vector containing the indexes as regards to the ECG signal
        # diff_flag is 1 if i want to get the diff vector in ms, otherwise just return the RR vector.
        RWL_all = []
        for i in range(0, len(signal['ecg']), 128):
            RWL = ast.literal_eval(signal['RWL'][i])
            for j in range(5):
                if RWL[j] != -1:
                    RWL_all.append(i + RWL[j])
        RWL_all = np.array(RWL_all)
        if diff_flag==0:
          return RWL_all
        return np.diff(1000*RWL_all/128) # difference in ms


#**********************************************************************************************************************

def HRV_Ext(signal, date):
    return pd.DataFrame({'nn50': pyhrv.time_domain.nn50(signal)[1],
             'sdnn': pyhrv.time_domain.sdnn(signal)[0],
             'sdann': pyhrv.time_domain.sdann(signal)[0],
             'sdsd': pyhrv.time_domain.sdsd(signal)[0],
             'TriIdx':pyhrv.time_domain.triangular_index(signal, plot=False, show=False)[0],
             'sdnnIdx':pyhrv.time_domain.sdnn_index(signal, full=True)[0]}, index=[date])

#**********************************************************************************************************************

def num_ecg(row):
    pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
    return [float(el) for el in pattern.findall(row['ECG'])]


#**********************************************************************************************************************

def get_ecg(path,end_time, start_delay=0, invert=False, unix=False):
  pd.options.mode.chained_assignment = None  # default='warn'
  # end_time is a float in units of hour - to set when to end the signal.
  # ecg_flag used to include or not include the ecg data.
  # start_delay is a float in units of hours - to set when to start the signal
  samples_end = round(end_time*60*60); samples_start = round(start_delay*60*60);
  try:
    patch = pd.read_csv(path, nrows=samples_start + samples_end)[samples_start:samples_start + samples_end]
  except Exception as e:
    return

  ecg = patch[['Record Time', 'RWL', 'Noise', 'ECG', 'Timezone', 'HR', 'RMSSD']]
  ecg.loc[:, 'ecg'] = ecg.apply(num_ecg, axis=1)
  
  ecg = (ecg
    .explode('ecg', ignore_index=True)
    .assign(ord=lambda x: x.groupby('Record Time').cumcount())
  )
  tdelta = pd.to_timedelta(1/128 * 1000, unit="ms")
  
  if invert:
    # Invert the signal if needed
    _, is_invert = nk.ecg_invert(np.array(ecg['ecg'][:128*60],dtype=np.float32), sampling_rate=128, show=False)
    ecg['ecg'] = -ecg['ecg'] if is_invert else ecg['ecg']
  
  if unix:
    ecg['unix'] = ecg['Record Time']
  
  ecg['Record Time'] = pd.to_datetime(ecg['Record Time'], unit='ms') + pd.to_timedelta(ecg['Timezone'], unit='seconds') + tdelta * ecg['ord']
  ecg.drop(['ECG', 'ord', 'Timezone'], axis=1, inplace=True)
  
  # if (invert) and (is_invert==1):
  #   return ecg, 'inverted'
  # else:
  return ecg

#**********************************************************************************************************************

def spectral_center(signal,Fs=128):
  # Calculate the Fourier Transform
  spectrum = np.fft.fft(signal)/len(signal)
  # Frequency bins
  freq_bins = np.fft.fftfreq(len(signal), d=1/Fs)
  # Keep only positive frequencies
  positive_freq_bins = freq_bins[:len(freq_bins)//2]
  positive_spectrum = spectrum[:len(freq_bins)//2]
  # Calculate the spectral centroid
  spectral_centroid = np.sum(positive_freq_bins * np.abs(positive_spectrum)) / np.sum(np.abs(positive_spectrum))
  return np.round(spectral_centroid,4)

#**********************************************************************************************************************
def min_max_norm(sig):
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig))
  
def norm(sig):
  if np.std(sig) == 0:
    return
  return (sig-np.mean(sig))/np.std(sig)

#**********************************************************************************************************************

def gini(x):
  total = 0
  for i, xi in enumerate(x[:-1], 1):
    total += np.sum(np.abs(xi - x[:]))
  return total / (2*(len(x)**2 * np.mean(x)))

#**********************************************************************************************************************

def ZCR(signal):
  mean_sig = np.mean(signal)
  zero_crossings = np.diff(np.sign(signal - mean_sig)).nonzero()[0]
  return round(len(zero_crossings)/len(signal)*100,3)

#**********************************************************************************************************************

def get_RR_segments(path, time_to_consider, filter_acc=True, invert=True):
  # Filter == True, means we'll do acc filtering as well to noise.
  
  acc, ecg_data = get_acc_ecg(path, time_to_consider, invert=invert)
  acc_time = acc['Record Time']
  ecg_time = ecg_data['Record Time']
  norm_mag = min_max_norm(acc['magnitude'])
  percentile_TOP = np.percentile(norm_mag[:25*60*60*4], 99.9)
  percentile_LOW = np.percentile(norm_mag[:25*60*60*4], 0.1)
  if ecg_data is None:
    return
  
  # For every place in ecg where there is Noise, change the value of ecg into -180
  ecg_data.loc[ecg_data['Noise'] == 1, 'ecg'] = -180
  rwl = np.array(RWL(ecg_data, 0), dtype=int); 
  ecg = np.array(ecg_data['ecg']);  
  Noise = np.array(ecg_data['Noise'])
  segments = {}; i = 0;
  
  for val in range(len(rwl)-1):
      start = rwl[val]; end = rwl[val+1]
      seg_N = Noise[start:end]
      L = len(seg_N)
      noise_ratio = 100 * np.sum(seg_N) / L
      if (filter_acc == False) and (noise_ratio > 10 or L > 120): 
        continue
      
      if filter_acc == True:  
        start_time = ecg_time[start]; end_time = ecg_time[end]
        start_idx = np.searchsorted(acc_time, start_time, side='left')
        end_idx = np.searchsorted(acc_time, end_time, side='right')
        acc_range = norm_mag[start_idx:end_idx]
        acc_within_range = (acc_range >= percentile_LOW) & (acc_range <= percentile_TOP)
        if (noise_ratio > 0) or (L > 120) or not np.all(acc_within_range): # in case of more than 10% Noise in the segment skip it
            continue
      
      segments[i] = ecg[start:end]
      i += 1      
  return segments

#**********************************************************************************************************************

def classify_segments(data, date):
    import matplotlib.pyplot as plt
    classifications = []
    ecg_segments = data[date]
    for i in range(len(ecg_segments)):
        segment = ecg_segments[i]

        plt.plot(segment)
        plt.title(f"{date}:     ECG Segment {i+1}")
        plt.xlabel('Time'); plt.ylabel('Amplitude')
        plt.show(block=True)  # Show the plot in a new window
        classification = input("0 - No TP, 1 - Low TP, 2 - High TP, b - break")
        plt.close()  # Close the plot window after input
        if classification.lower() == 'b':
            print("Classification process stopped.")
            return None
        elif classification.lower() == 's': # skip
            classifications.append(-100)
        elif classification == '0':
            classifications.append(0)
        elif classification == '1':
            classifications.append(1)
        elif classification == '2':
            classifications.append(2)
    
    return classifications

#**********************************************************************************************************************

class ECGTPDetector(nn.Module):
    def __init__(self, num_classes=3):
        super(ECGTPDetector, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM layer
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape input to (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        
        # LSTM layer
        x, _ = self.lstm(x.transpose(1, 2))
        x = x[:, -1, :]  # Get the last output from the LSTM
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
      
#**********************************************************************************************************************

def get_active_times(path):
    # this function outputs a dictionary of active times
    # each segment in the dict has start and end times.
    
    acc = pd.read_csv(path)
    acc['record time'] = pd.to_datetime(acc['record time'])
    acc['accind'] = acc.index
    acc['activity_diff'] = acc['activity'].diff()
    
    active_times = []
    for seg_start_ind in acc[acc.activity_lens >= 60 / 60].index[:]:
        seg_start_time = acc.loc[seg_start_ind, 'record time']
        seg_end_ind = acc[(acc['record time'] >= seg_start_time) & (acc['activity_diff'] == -1)].index[0]-1
        seg_end_time = acc.loc[seg_end_ind, 'record time']
        active_times.append((seg_start_time, seg_end_time))
    return active_times

#**********************************************************************************************************************

def get_ecg_active(ecg_path, acc_path):
    active_times = get_active_times(acc_path)
    ecg_data = get_ecg(ecg_path,24)

    ecg_active = None
    for i in range(len(active_times)):
        start_time = active_times[i][0]; end_time = active_times[i][1]; 
        temp = ecg_data[(ecg_data['Record Time'] > start_time) & (ecg_data['Record Time'] < end_time)]
        ecg_active = pd.concat([ecg_active, temp], ignore_index=True)
        
    return ecg_active
  
#********************************************************************************************************************** 
  
def parallel_ExtractActive(date):
  date_acc = date[:4] + date[5:7] + date[8:10]
  acc_path = "/newvolume/trial_II_files/AY03002/accs"
  acc_path = acc_path + f'/acc_{date_acc}.csv'
  ecg_path = "/newvolume/trial_II_files/AY03002/patch_files/"
  ecg_path = ecg_path + f'ECGRec_202405_E117849_{date}_ECG_denoised.csv'
  try:
    ecg_active = get_ecg_active(ecg_path, acc_path)
    path = '/newvolume/Nadav/AY03002/Active_ecg' + f'/ecg_active_{date}.pkl'
    with open(path, 'wb') as f:
          pickle.dump(ecg_active, f)
  except Exception as e:
    print(e)

#**********************************************************************************************************************                 
        
def get_ecg_Nonactive(ecg_path, acc_path):
    active_times = get_active_times(acc_path)
    ecg_data = get_ecg(ecg_path,24)

    ecg_Nonactive = None
    if not active_times:
        print('No Active times')
        return ecg_data
    
    temp = ecg_data[(ecg_data['Record Time'] < (active_times[0][0]))]
    ecg_Nonactive = pd.concat([ecg_Nonactive, temp], ignore_index=True)
    
    i = -1
    for i in range(len(active_times)-1):
        end_first = (active_times[i][1]); start_second = (active_times[i+1][0]); 
        temp = ecg_data[(ecg_data['Record Time'] > end_first) & (ecg_data['Record Time'] < start_second)]
        ecg_Nonactive = pd.concat([ecg_Nonactive, temp], ignore_index=True)
        
    temp = ecg_data[(ecg_data['Record Time'] > (active_times[i+1][1]))]
    ecg_Nonactive = pd.concat([ecg_Nonactive, temp], ignore_index=True)
        
    return ecg_Nonactive

#********************************************************************************************************************** 

def parallel_ExtractNonActive(date):
  date_acc = date[:4] + date[5:7] + date[8:10]
  acc_path = "/newvolume/trial_II_files/AY03002/accs"
  acc_path = acc_path + f'/acc_{date_acc}.csv'
  ecg_path = "/newvolume/trial_II_files/AY03002/patch_files/"
  ecg_path = ecg_path + f'ECGRec_202405_E117849_{date}_ECG_denoised.csv'
  try:
    ecg_Nonactive = get_ecg_Nonactive(ecg_path, acc_path)
    path = '/newvolume/Nadav/AY03002/NonActive_ecg' + f'/ecg_Nonactive_{date}.pkl'
    with open(path, 'wb') as f:
          pickle.dump(ecg_Nonactive, f)
  except Exception as e:
    print(e)

#********************************************************************************************************************** 
        
def get_PR_parallel(input):
  path = input[0]; date = input[1]
  try:
    # with open(path, 'rb') as f:     # in case we work with pickle files
    #     patch = pickle.load(f)
    
    patch = pd.read_csv(path, nrows=24*60*60)          # in case we work with csv files
    PR_data = patch[['Record Time', 'HR', 'RMSSD']]
    
    # Calculations
    PR = PR_data['HR']
    RMSSD = PR_data['RMSSD']
    errors_pre = np.sum((PR < 40) | (PR > 255))/len(PR)*100

    #Remove PR Errors:
    PR = PR[(PR > 40) & (PR < 255)]

    data_all = {'Date': date, 'Mean - PR': np.round(np.mean(PR),3),
            'STD - PR':np.round(np.std(PR),3), 'Median - PR':np.round(np.median(PR),3),
            'Unique Count - PR':len(set(PR)),  'Number of samples - PR':len(PR),
            'ZCR - PR': ZCR(PR), '% of Errors-PR Reading - PR': np.round(errors_pre,2),
            'Mean - RMSSD': np.round(np.mean(RMSSD),3),
            'STD - RMSSD':np.round(np.std(RMSSD),3), 'Median - RMSSD':np.round(np.median(RMSSD),3),
            'Unique Count - RMSSD':len(set(RMSSD)),  'ZCR - RMSSD': ZCR(RMSSD)}
    return data_all
  except Exception as e:
    print(f'{date}  {e}')
    return {'Date': date, 'Mean - PR': None,'STD - PR': None, 'Median - PR':None,
            'Unique Count - PR':None,  'Number of samples - PR':None,
            'ZCR - PR': None, '% of Errors-PR Reading - PR': None,
            'Mean - RMSSD': None,  'STD - RMSSD':None, 'Median - RMSSD':None,
            'Unique Count - RMSSD':None,  'ZCR - RMSSD': None}

#********************************************************************************************************************** 

def get_TP_scores_NN(path, hours=24):
  try:
    temp = get_RR_segments(path, hours, invert=True)         # in case we work with csv files
    # temp = get_RR_segments_parallel(path)     # in case we work with pickle files
    if not temp:
      return None
    model = ECGTPDetector(num_classes=3)
    model.load_state_dict(torch.load("TP_NN_model.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    ecg_segments_list = []; 

    for seg in range(len(temp)):
        ecg_segment = (temp[seg])
        ecg_segments_list.append(ecg_segment)

    max_length = 120 # pad to length of 120

    # Pad each segment with zeros to match the maximum length
    padded_segments = []
    for seg in ecg_segments_list:
        padded_seg = np.pad(seg, (0, max_length - len(seg)), mode='constant')
        padded_segments.append(np.array(padded_seg, dtype=float))
        
    ecg_segments_tensor = torch.tensor(np.array(padded_segments), dtype=torch.float32)

    with torch.no_grad():
        outputs = model(ecg_segments_tensor)
        _, test_predicted = torch.max(outputs, 1)
    test_predicted = test_predicted.numpy()
    
    res = {
        "NN_mean": np.mean(test_predicted),
        "TP_NN_median": np.median(test_predicted),
        "TP_NN_std": np.std(test_predicted),
        "TP_NN_skewness": skew(test_predicted),
        "TP_NN_kurtosis": kurtosis(test_predicted)
    }

    return res
  
  except Exception as e:
      print(f"Error processing {path}: {e}")
      return None
#********************************************************************************************************************** 

def get_RR_segments_parallel(path, num_segs = 1000):
  try:
    data = get_ecg(path,24, invert=True)
    # with open(path, 'rb') as f:
    #     data = pickle.load(f)
    # For every place in ecg where there is Noise, change the value of ecg into -180
    data.loc[data['Noise'] == 1, 'ecg'] = -180
    rwl = np.array(RWL(data, 0), dtype=int); ecg = (data['ecg']);
    segments = {}; i = 0;
    
    if num_segs is None:
      num_segs = len(rwl)-1
      
    for val in range(len(rwl)-1):
        start = (rwl[val]); end = (rwl[val+1])
        seg = data[start:end]
        L = len(seg)
        if L == 0:
          continue
        noise_ratio = 100 * np.sum(seg['Noise']) / L
        if noise_ratio > 10 or L > 120: # in case of more than 10% Noise in the segment skip it
            continue
        segments[i] = ecg[start:end]
        
        if i == num_segs:
          return segments
        
        i += 1
    return segments
  except Exception as e:
    print(f"Error processing {path}: {e}")

#********************************************************************************************************************** 

# Including gardient calculation and median to minmum distance.
def T_P_check_parallel(path, hours=24):
    try:
        # with open(path, 'rb') as f:     # in case we work with pickle files
        #     data = pickle.load(f)   
        data = get_ecg(path, hours)          # in case we work with csv files
        
        # For every place in ecg where there is Noise, change the value of ecg into -180
        data.loc[data['Noise'] == 1, 'ecg'] = -180
        diff = 10; # This value will be used to differentiate between the QRS and the TP
        rwl = np.array(RWL(data, 0)); ecg = np.array(norm(data['ecg']));
        gardient=[]; min_median = []
        for val in range(len(rwl)-1):
            start = int(rwl[val]); end = int(rwl[val+1])
            segment_max = ecg[start+diff:end-3*diff] # where to look for maximum between RR peaks

            if segment_max.size == 0: # any case where rwl is wrong
                continue
            
            max_idx = np.argmax(segment_max) + start + diff
            segment_min = ecg[start+diff:max_idx] # where to look for minimum between RR peaks
            if segment_min.size == 0: # any case where rwl is wrong
                continue
            
            dy = np.min(segment_min) - np.max(segment_max)
            dt = np.argmin(segment_min) + start + diff - max_idx
            
            if dt != 0: # this happens where the ecg is flat, so when there is Noise
                gardient.append(dy / dt)    
                min_median.append(np.median(segment_max) - np.min(segment_min))
        return trim_mean(gardient, 0.1), trim_mean(min_median, 0.1) # using trim removes the 10% outliers from the smallest and largest values
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

########################################################
def T_P_check_parallel_filter_acc(path, hours=24):
    try:
        # with open(path, 'rb') as f:     # in case we work with pickle files
        #     data = pickle.load(f)   
        acc, data_ecg = get_acc_ecg(path, hours, invert=True)          # in case we work with csv files
        
        acc_time = acc['Record Time']
        ecg_time = data_ecg['Record Time']
        norm_mag = min_max_norm(acc['magnitude'])
        percentile_TOP = np.percentile(norm_mag[:25*60*60*4], 99.9)
        percentile_LOW = np.percentile(norm_mag[:25*60*60*4], 0.1)

        # For every place in ecg where there is Noise, change the value of ecg into -180
        data_ecg.loc[data_ecg['Noise'] == 1, 'ecg'] = -180
        diff = 10; # This value will be used to differentiate between the QRS and the TP
        rwl = np.array(RWL(data_ecg, 0)); ecg = np.array(norm(data_ecg['ecg']));
        Noise = np.array(data_ecg['Noise'])
        gardient=[]; min_median = []
        for val in range(len(rwl)-1):
            start = int(rwl[val]); end = int(rwl[val+1])
            
            # Noise Check
            start_time = ecg_time[start]; end_time = ecg_time[end]
            start_idx = np.searchsorted(acc_time, start_time, side='left')
            end_idx = np.searchsorted(acc_time, end_time, side='right')
            acc_within_range = (norm_mag[start_idx:end_idx] >= percentile_LOW) & (norm_mag[start_idx:end_idx] <= percentile_TOP)
            seg_N = Noise[start:end]
            L = len(seg_N)
            noise_ratio = 100 * np.sum(seg_N) / L
            if (noise_ratio > 0) or (L > 120) or (~np.all(acc_within_range)): # in case of more than 10% Noise in the segment skip it
                continue
            
            # Calculation
            segment_max = ecg[start+diff:end-3*diff] # where to look for maximum between RR peaks
            if segment_max.size == 0: # any case where rwl is wrong
                continue
            
            max_idx = np.argmax(segment_max) + start + diff
            segment_min = ecg[start+diff:max_idx] # where to look for minimum between RR peaks
            if segment_min.size == 0: # any case where rwl is wrong
                continue
            
            dy = np.min(segment_min) - np.max(segment_max)
            dt = np.argmin(segment_min) + start + diff - max_idx
            
            if dt != 0: # this happens where the ecg is flat, so when there is Noise
                gardient.append(dy / dt)    
                min_median.append(np.median(segment_max) - np.min(segment_min))
                
        result = {
            "TP_gradient_mean": trim_mean(gardient, 0.1),
            "TP_gradient_median": np.median(gardient),
            "TP_gradient_std_dev": np.std(gardient),
            "TP_gradient_skewness": skew(gardient),
            "TP_gradient_kurtosis": kurtosis(gardient),
            "TP_min_median_mean": trim_mean(min_median, 0.1),
            "TP_min_median_median": np.median(min_median),
            "TP_min_median_std_dev": np.std(min_median),
            "TP_min_median_skewness": skew(min_median),
            "TP_min_median_kurtosis": kurtosis(min_median),
        }
        
        return result
    
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None      
#********************************************************************************************************************** 
    
def get_SpO2(path):
  try:
    df = pd.read_csv(path)
  except FileNotFoundError:
    return None
  keys_we_need = ['Record Time', 'SpO2', 'PR']
  SpO2 = df[keys_we_need]
#   SpO2 = (SpO2
#   .explode('SpO2', ignore_index=True)
#   .assign(ord=lambda x: x.groupby('Record Time').cumcount())
# )
#   tdelta = pd.to_timedelta(1/2 * 1000, unit="ms")
  # SpO2['Record Time'] = SpO2['Record Time'].apply(lambda x: pd.Timestamp(x, unit='ms')) + pd.to_timedelta(2, unit="H") + tdelta * SpO2['ord']
  SpO2.loc[:, 'Record Time'] = SpO2['Record Time'].apply(lambda x: pd.Timestamp(x, unit='ms')) + pd.to_timedelta(2, unit="H")
  return SpO2

#********************************************************************************************************************** 

def get_ecg_precentile(path,end_time, start_delay=0, diff_flag=1):
  # end_time is a float in units of hour - to set when to end the signal.
  # ecg_flag used to include or not include the ecg data.
  # start_delay is a float in units of hours - to set when to start the signal
  samples_end = round(end_time*60*60); samples_start = round(start_delay*60*60);
  try:
    patch = pd.read_csv(path, nrows=samples_end)[samples_start:samples_start + samples_end]
  except Exception as e:
    return

  # ecg______________________________________________________________________
  ecg = patch.drop(['App ID', 'Sensor SN', 'Sensor MAC', 'Collect Time',
       'Receive Time', 'Sensor Info',
       'SDK Version', 'Flash', 'Lead On', 'Activity', 'Magnification',
       'Acc Accuracy', 'Sample Frequency',
       'posture', 'Acc Activity', 'Acc Step Offset',
       'Acc Step Total', 'Temperature', 'EEAlgo'], axis=1)

  ecg['ecg'] = ecg.apply(num_ecg, axis=1)
  ecg = (ecg
    .explode('ecg', ignore_index=True)
    .assign(ord=lambda x: x.groupby('Record Time').cumcount())
  )
  tdelta = pd.to_timedelta(1/128 * 1000, unit="ms")
  ecg['Record Time'] = pd.to_datetime(ecg['Record Time'], unit='ms') + pd.to_timedelta(ecg['Timezone'], unit='seconds') + tdelta * ecg['ord']
  ecg.drop(['ACC', 'ECG', 'ord', 'Timezone'], axis=1, inplace=True)

  rwl = RWL(ecg, diff_flag)
  return rwl, ecg

#********************************************************************************************************************** 

def get_ecg_precentile_parallel(path):
    try:
        rwl, ecg = get_ecg_precentile(path, end_time=24, start_delay=0, diff_flag=0) # from csv file
        
        RR_peaks = ecg['ecg'][rwl]
        percentile_list = []
        for per in [10,20,30,40,60]:
            percentile_list.append(np.percentile(RR_peaks, per))
    except Exception as e:
        print(e)        
        return
    return percentile_list
 
 #********************************************************************************************************************** 

def plot_colors(dates):
  colors = [plt.cm.get_cmap('Set1')(nn) if nn != 5 else plt.cm.get_cmap('Set2')(nn) for nn in [0, 4, 5, 2, 1, 3, 8, 6, 7]]

  day_colors = {
    0: colors[1],     # Monday
    1: colors[2],     # Tuesday
    2: colors[3],     # Wednesday
    3: colors[4],     # Thursday
    4: colors[5],     # Friday
    5: colors[6],     # Saturday
    6: colors[0]      # Sunday
  }
  # Create legend
  day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
  legend_handles = [mpatches.Patch(color=day_colors[i], label=day_labels[i]) for i in range(7)]
  plt.legend(handles=legend_handles, title="Days", ncol=7, loc='upper left', bbox_to_anchor=(0,1.8),fontsize='small', 
             title_fontsize='small', handletextpad=0.4, columnspacing=0.5, borderpad=0.3)

  
  
  
  # Function to get the day of the week for a given date
  def get_day_of_week(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    return date.weekday()
  # Map each date to the relevant color based on the day of the week
  colors_mapped = [day_colors[get_day_of_week(date)] for date in dates]
  return colors_mapped

#********************************************************************************************************************** 

def plot_events(ax, dates, events, height_annotate=1, y_vals=None, red_line_height=0.1):
    # ax - the axis where to plot the events.
    
    # you can choose whether to anntotate the events on a specific height: y_to_annotate,
    # or on the exact Y-value of the specific height
    # height_annotate is the Y - height you want to annotate the events
    # y_vals are the Y-values of the data you want to annotate on
    # red_line_height is the factor to which to plot the red line in the graph
    
    if y_vals:
        annotations_height = y_vals
    else:
        annotations_height = height_annotate * np.ones(len(dates))
        
    line = np.zeros(len(dates)); ticks_color = [None]*len(dates); i=0
    prev_event_end = None
    for date, y_val in zip(dates, annotations_height):
        event = events[date]
        
        if event:
          if event[:8] == '+MIDDLE_':
            ax.annotate(event[8:-2], xy=(date, y_val-0.3), xytext=(0, 0),
                textcoords='offset points', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'), rotation='vertical', size=8)
          else:
            ax.annotate(event[1:-2], xy=(date, y_val), xytext=(0, 0),
                            textcoords='offset points', ha='center', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'), rotation='vertical', size=8)
          
        # Build the line
        if event is None:
            line[i] =  -500
        elif event[0] == '-':
            line[i] =  -500
        else:
            line[i] = y_val + red_line_height
        # Build the ticks color
        red_color = plt.cm.get_cmap('Set1')(0)  # RGBA value
        pink_color = plt.cm.get_cmap('Dark2')(3)
        if event == None:
          ticks_color[i] = 'black'
        elif event[-1] == 'M':
          ticks_color[i] = pink_color
        elif event[-1] == 'm':
          ticks_color[i] = red_color
        i += 1
        
            
    ax.scatter(dates,line, color='red',linewidth=1)
    for tick, color in zip(ax.get_xticklabels(), ticks_color):
      tick.set_color(color)
      if color != 'black':
          tick.set_fontweight('bold')
#********************************************************************************************************************** 

def Kmeans_plot(PR_data, ID, dates, events):
  #K-means over whole day
  num_clusters = 4
  # Perform K-means clustering
  kmeans = KMeans(n_clusters=num_clusters, n_init=100, random_state=3)
  dates_add = (set(PR_data.index)- set(PR_data.dropna().index))
  PR_data = PR_data.dropna()
  df_all_PR_labels = PR_data.loc[:, ['Mean - PR']]

  kmeans.fit(df_all_PR_labels)
  labels = kmeans.labels_

  labels = kmeans.predict(df_all_PR_labels)
  PR_data['Labels'] = labels
  
  new_rows = [pd.DataFrame([[None] * 13], index=[date], columns=PR_data.columns) for date in dates_add]
  PR_data = pd.concat([PR_data] +new_rows)
  PR_data = PR_data.sort_index()
  
  fig,ax = plt.subplots(figsize=(35,3))
  ax.plot(dates, list(PR_data.Labels))

  plot_events(ax, dates, events, height_annotate=3.15, red_line_height=0.5)

  plt.yticks([0.0,1.0,2.0,3.0]); 
  plt.title(f"{ID} - ECG Kmeans Over 24H\n\n")
  plt.xlabel('Dates')
  plt.ylabel('Cluster')
  plt.xticks(rotation='vertical')
  plt.xlim(dates[0], dates[-1])
  plt.ylim(0,4); plt.ylabel('Cluster')
  plt.grid(axis='y')
  ax.vlines(ax.get_xticks(), 0, 3, colors='gray', linestyles='-', linewidth=0.5)
  plt.show()




#**********************************************************************************************************************
# Vladimir's Original Functions
def acc_abs(row):
  nums = [float(el) for el in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", row['ACC'])]
  vecs = [nums[3 * i:3 * (i + 1)] for i in range(len(nums) // 3)]
  return vecs

def get_acc_ecg(path, end_time=4, start_delay=0, invert=False, unix=False):
  pd.options.mode.chained_assignment = None  # default='warn'
  # end_time is a float in units of hour - to set when to end the signal.
  # ecg_flag used to include or not include the ecg data.
  # start_delay is a float in units of hours - to set when to start the signal
  samples_end = round(end_time*60*60); samples_start = round(start_delay*60*60);
  try:
    patch = pd.read_csv(path, nrows=samples_start + samples_end)[samples_start:samples_start + samples_end]
  except Exception as e:
    return None, None
# _________________________________________________ ECG __________________________________________
  ecg = patch[['Record Time', 'RWL', 'Noise', 'ECG', 'Timezone', 'HR', 'RMSSD', 'SNR']]
  ecg.loc[:, 'ecg'] = ecg.apply(num_ecg, axis=1)
  
  ecg = (ecg
    .explode('ecg', ignore_index=True)
    .assign(ord=lambda x: x.groupby('Record Time').cumcount())
  )
  tdelta = pd.to_timedelta(1/128 * 1000, unit="ms")
  # Invert the signal if needed
  if invert == True:
    _, is_invert = nk.ecg_invert(np.array(ecg['ecg'],dtype=np.float32), sampling_rate=128, show=False)
    ecg['ecg'] = -ecg['ecg'] if is_invert else ecg['ecg']

  if unix:
    ecg['unix'] = ecg['Record Time']
  ecg['Record Time'] = pd.to_datetime(ecg['Record Time'], unit='ms') + pd.to_timedelta(ecg['Timezone'], unit='seconds') + tdelta * ecg['ord']
  ecg.drop(['ECG', 'ord', 'Timezone'], axis=1, inplace=True)
  
# _________________________________________________ ACC __________________________________________
  acc = patch[['Record Time', 'Timezone', 'ACC']]
  acc['acc'] = acc.apply(acc_abs, axis=1)
  acc = (acc
    .explode('acc', ignore_index=True)
    .assign(ord=lambda x: x.groupby('Record Time').cumcount())
  )
  acc[['x', 'y', 'z']] = pd.DataFrame(acc.acc.tolist(), index=acc.index)
  acc['magnitude'] = acc['acc'].apply(lambda x: np.linalg.norm(x))
  tdelta = pd.to_timedelta(1/25 * 1000, unit="ms")
  acc['Record Time'] = pd.to_datetime(acc['Record Time'], unit='ms') + pd.to_timedelta(acc['Timezone'], unit='seconds') + tdelta * acc['ord']
  acc.drop(['ACC', 'ord', 'acc'], axis=1, inplace=True)  
  return acc, ecg



#**********************************************************************************************************************
################################################### QC ################################################################

def NoiseFilter(path, hours=4, invert=True):
  try:
    # Filter - Choose only segments with no noise around them both in Noise vector and in ACC 
    noise_time_thresh = 10               # how much time before and after the relevant one minute segment, to look for noise?
    segment_duration = 60                # Duration of ECG segment to be added to valid_segments
    Fs = 128
    valid_segments = np.empty((0,Fs*segment_duration))
    acc_bef, ecg_bef = get_acc_ecg(path, hours, invert=invert)

    # Make sure signals are during night times before 6AM
    ecg_bef = ecg_bef.loc[ecg_bef['Record Time'].dt.hour < 6]
    acc_bef = acc_bef.loc[acc_bef['Record Time'].dt.hour < 6]
    
    if ecg_bef.shape[0] == 0:
      return
    
    norm_mag = min_max_norm(acc_bef['magnitude'])
    percentile_TOP = np.percentile(norm_mag, 99.9)
    percentile_LOW = np.percentile(norm_mag, 0.1)

    ecg_values = ecg_bef['ecg'].values
    noise_values = ecg_bef['Noise'].values
    ecg_time = ecg_bef['Record Time']
    acc_time = acc_bef['Record Time']

    for i in range(Fs*noise_time_thresh, ecg_bef.shape[0] - Fs*60*1, Fs*60*1): # start one minute after record started, step: every five minutes, untill the end of ECG record
        noise_segment = noise_values[i-Fs*noise_time_thresh : i+Fs*(noise_time_thresh+segment_duration)] #one minutes before and after segment
        ecg_segment = ecg_values[i : i + Fs * segment_duration]
        
        if np.sum(noise_segment) == 0:                 # Check no noise appearence one minutes before and after segment
            start_time = ecg_time[i - Fs * noise_time_thresh]
            end_time = ecg_time[i + Fs * (noise_time_thresh + segment_duration) - 1]
            acc_mask = (acc_time >= start_time) & (acc_time <= end_time)
            acc_seg = norm_mag[acc_mask]
            acc_within_range = (acc_seg >= percentile_LOW) & (acc_seg <= percentile_TOP)
            
            if np.all(acc_within_range):                      # Check that no movement is done one minutes before and after segment             
                valid_segments = np.vstack([valid_segments, ecg_segment])
                # times_of_valid.append(ecg_time[i])
  except Exception as e:   
    print(e)
    return 
  return valid_segments