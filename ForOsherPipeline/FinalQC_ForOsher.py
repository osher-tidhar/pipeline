import neurokit2 as nk
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import RobustScaler
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
import numpy as np
from googleapiclient.http import MediaFileUpload
import boto3
from tqdm import tqdm
import traceback

# TODO change base_path in PatientInfo
from PatientsInfo import *
from functions import get_acc_ecg, min_max_norm
from QC_Alerts import *

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")

"""In regards to this error:
/newvolume/Nadav/QC/ForOsherPipeline/FinalQC_ForOsher.py:421: FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated. In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes. To retain the old behavior, exclude the relevant entries before the concat operation.
  combined_data = pd.concat([new_patient_data, old_patient_data], ignore_index=True)  # Add new data on top
Ignore it below """
warnings.simplefilter(action='ignore', category=FutureWarning)

# TODO: Send as an argument OR as a global variable
SEND_QC_ALERTS = False
#SEND_QC_ALERTS = True

# Function to upload file to Google Drive
def upload_file_to_drive(local_file_path, upload_file_name=None, credentials_path=base_path + '/credentials.json'):
    """
    Uploads a file from the local machine to Google Drive with a specified name.

    :param local_file_path: str, the path to the local file to be uploaded.
    :param upload_file_name: str, the desired name of the file on Google Drive (optional).
    :param credentials_path: str, path to the service account credentials file.
    """
    drive_folder_id = '1phoMnVXwDcXWiDbgClXh0MC5yMQCWRNR'  # Replace with your Google Drive folder ID

    # Authenticate Google Drive API
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    # Use the provided name or default to the original file name
    file_name_on_drive = upload_file_name if upload_file_name else os.path.basename(local_file_path)

    # Determine the MIME type based on the file extension
    _, file_extension = os.path.splitext(local_file_path)
    if file_extension == '.csv':
        mime_type = 'text/csv'
    elif file_extension in ['.xlsx', '.xls']:
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    else:
        mime_type = 'application/octet-stream'  # Default for unknown file types

    # Upload the file to Google Drive
    file_metadata = {
        'name': file_name_on_drive,  # The name of the file on Google Drive
        'parents': [drive_folder_id]  # The folder where the file will be uploaded
    }
    media = MediaFileUpload(local_file_path, mimetype=mime_type)

    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    
    print(f"File uploaded to Google Drive with file ID: {uploaded_file.get('id')}")

def get_active_patients():
    """
    This function retrieves active patients and their latest sensor information
    Returns two dictionaries:
    active_patients_Spo2: Maps patient IDs to their latest SpO2 sensor
    active_patients_ECG: Maps patient IDs to their latest ECG sensor
    Only includes patients who have not ended their trial (trial_end is NaN)
    """
    active_patients_ECG = {}; active_patients_Spo2 = {}
    patients = get_patients_info()
    PatientsID = list(patients['Patient ID'])

    for patient in PatientsID:
        trial_end = patients[patients['Patient ID'] == patient]['Trial end'].iloc[0]
        first_visit = patients[patients['Patient ID'] == patient]['1st Visit'].iloc[0]

        if trial_end == 'out' or first_visit == 'out' or str(trial_end) != 'nan':
            continue
        patient_info = Patient(patient)
        active_patients_ECG[patient] = patient_info.sensors[-1][0]
        active_patients_Spo2[patient] = patient_info.SpO2[-1]
    return active_patients_Spo2, active_patients_ECG

def get_QRS(all_valid):
    """
    Function to extract and process QRS complexes from ECG segments
    Input: all_valid - array of valid ECG segments
    Output: mean_ecgs - array of averaged QRS complexes for each segment
    """
    # Set fixed length for QRS complex interpolation
    ecg_maxlen = 120
    mean_ecgs = np.empty((all_valid.shape[0], ecg_maxlen))
    # Process each ECG segment
    for row, seg in enumerate(all_valid):
        seg = np.array(seg,dtype=np.float32)
        # Clean ECG signal using neurokit2
        try:
            
            seg_clean = nk.ecg_clean(seg, sampling_rate=128)
            # Segment ECG into individual QRS complexes
            qrs_epochs = nk.ecg_segment(seg_clean, rpeaks=None, sampling_rate=128, show=False)
        except:
            continue
        ecgs = np.empty((ecg_maxlen, len(qrs_epochs)))

        # Process each QRS complex
        for i, key in enumerate(qrs_epochs.keys()):
            # Extract QRS values
            org_qrs = qrs_epochs[key]['Signal'].values
            # Interpolate QRS to fixed length of 120 samples
            intrep_qrs = interp_signal(org_qrs, ecg_maxlen)   
            ecgs[:,i] =  intrep_qrs 
        # Calculate mean QRS complex for this segment
        mean_QRS = np.nanmean(ecgs,axis=1)                   
        mean_ecgs[row,:] = mean_QRS
    return mean_ecgs

def NoiseFilter(path):
    """
    Filters ECG segments to find valid segments with no noise or movement artifacts.
    
    This function analyzes ECG and accelerometer data to identify clean ECG segments that:
    1. Have no noise in the ECG signal before and after the segment
    2. Have minimal movement (accelerometer magnitude within 0.1-99.9 percentile range)
    3. Were recorded during nighttime hours (before 6 AM)
    
    Args:
        path (str): Path to the ECG data file
        
    Returns:
        dict: Dictionary with date as key and array of valid ECG segments as value
    """
    try:
        # Filter - Choose only segments with no noise around them both in Noise vector and in ACC 
        noise_time_thresh = 60               # how much time before and after the relevant one minute segment, to look for noise?
        segment_duration = 10                # Duration of ECG segment to be added to valid_segments
        all_valid = {}
        Fs = 128
        date = path.split('/')[-2]
        valid_segments = np.empty((0,Fs*segment_duration))
        acc_bef, ecg_bef = get_acc_ecg(path, 4, invert=False)
            
        if ecg_bef is None:
            print(f"No ECG data found in: {path.split('/')[-1]}")
            return
        
        # Make sure signals are during night times before 6AM
        ecg_bef = ecg_bef.loc[ecg_bef['Record Time'].dt.hour < 6]
        acc_bef = acc_bef.loc[acc_bef['Record Time'].dt.hour < 6]
            
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
        
        all_valid[date] = valid_segments 
        return all_valid
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        filename, line, _, _ = tb[-1]  # Get the last (innermost) frame
        print(f"NoiseFilter failed: {path.split('/')[-1]}: {e} (Line: {line}, File: {filename})")

def get_mean_QRSs(path):
    """
    Calculates mean QRS complexes from valid ECG segments in a file
    
    This function:
    1. Filters ECG data to get valid segments using NoiseFilter()
    2. Extracts and processes QRS complexes from valid segments using get_QRS()
    3. Returns array of mean QRS complexes
    
    Args:
        path (str): Path to the ECG data file
        
    Returns:
        numpy.ndarray: Array of mean QRS complexes, or None if error occurs
    """
    try:
        all_valid = NoiseFilter(path)
        date = path.split('/')[-2]
        if all_valid is None:
            return
        mean_QRS = get_QRS(all_valid[date])
        return mean_QRS
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        filename, line, _, _ = tb[-1]  # Get the last (innermost) frame
        print(f"get_mean_QRSs failed: {path.split('/')[-1]}: {e} (Line: {line}, File: {filename})")
        return

def interp_signal(signal, pad_len=120):
    """
    Interpolates a signal to a fixed length using quadratic interpolation
    
    Args:
        signal (array-like): Input signal to interpolate
        pad_len (int): Desired length of output signal (default 120)
        
    Returns:
        array: Interpolated signal of length pad_len
    """
    n = len(signal)
    x_old = np.arange(n)
    f = interp1d(x_old, signal, fill_value="extrapolate", kind='quadratic')  # Create interpolation function
    x_new = np.linspace(0, n - 1, pad_len)
    return f(x_new)

def Discontinuity_Noise_check(path):
    """
    Checks ECG data for discontinuities and noise levels
    
    This function analyzes ECG data to:
    1. Only considers data from before 6 AM
    2. Count number of discontinuities (gaps > 10 seconds) in the data
    
    Args:
        path (str): Path to the ECG data file
        
    Returns:
        tuple: (noise_percentage, num_discontinuities)
        None: If there is an error reading or processing the file
    """
    # Check for discontinuity and noise in the data
    # If there is a discontinuity in the data, the function will return number of discontinuities
    try:
        temp = pd.read_csv(path, nrows = 60*60*4)
        record_time = pd.to_datetime(temp['Record Time'] + temp['Timezone'][0]*1000, unit='ms')
        temp = temp.loc[record_time.dt.hour < 6]
        if len(temp['Noise']) == 0:
            print(f"No ECG data found in NIGHT: {path.split('/')[-1]}")
            return None, None
        noise = np.round((100 * np.sum(temp['Noise']) / len(temp['Noise'])),1)
        discon = np.count_nonzero(temp['Record Time'].diff() > 10000)  # 10000 is 10 second in ms
        return noise, discon
    
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        filename, line, _, _ = tb[-1]  # Get the last (innermost) frame
        print(f"Discontinuity_Noise_check failed {path.split('/')[-1]}: {e} (Line: {line}, File: {filename})")
        return 
    
def DTW_calc(QRSs_dict, patient_old_data=None):
    """
    Calculate DTW between each consecutive segments in a day. Get mean and std of the DTW calculations. 
    Take one mean QRS from each date. Calculate it's DTW with the next day mean. compare to thresh.
    :param QRSs_dict: dictionary of QRSs segments from each date
    :return: good_day: dictionary of good/bad days. 
    """
    try:
        if (patient_old_data is not None) and str(patient_old_data.Last_date) != 'nan':
            distances_old = np.fromstring(patient_old_data.Distances_new.strip('[]'), sep=' ')
            mean_qrs_old = np.fromstring(patient_old_data.Mean_qrs_new.strip('[]'), sep=' ')
            last_condition = patient_old_data.Condition
            first_date = 0              # flag for the starting date, where we want to build the distances_old, mean_qrs_old for the first time
        else:
            first_date = 1
            
        distances_new = None
        mean_qrs_new = None
            
        good_day = {}

    
        if (QRSs_dict is None) or (QRSs_dict.shape[0] <= 1): # in case one or less segments were found in this date, it means probably there is to much noise in this date - mark as -1
            good_day = None
            return good_day, distances_new, mean_qrs_new
        
        num_seg = QRSs_dict.shape[0]-1
        scaler = RobustScaler()
        
        norm_qrs_date = scaler.fit_transform(QRSs_dict.T).T
        mean_qrs = np.mean(norm_qrs_date,axis=0)                    # get the mean QRS out of all segments from the date
        
        distances = np.zeros(num_seg); 
        
        for i in range(num_seg):
            sig1 = norm_qrs_date[i].reshape(-1,1); sig2 = norm_qrs_date[i+1].reshape(-1,1)
            distances[i], _ = fastdtw(sig1, sig2, dist=euclidean)       # Calculate the distances for each consecutive segments
            
        if first_date == 1:
            distances_new = distances
            mean_qrs_new = mean_qrs           
            good_day = None
            return good_day, distances_new, mean_qrs_new
        else:
            mean_qrs_new = mean_qrs
            distances_new = distances
        
        divide = np.percentile(distances_new, 30) / 15
        thresh = ( np.mean(distances_new) + np.std(distances_new) + np.mean(distances_old) + np.std(distances_old) ) / divide
        
        sig1 = mean_qrs_old.reshape(-1,1); sig2 = mean_qrs_new.reshape(-1,1)
        dist_old_new, _ = fastdtw(sig1, sig2, dist=euclidean)
        if dist_old_new <= thresh:      # if the distance of mean QRS from yesterday to today is smaller than the thresh, we have a day where no noise and good wear of patch  
            good_day = 1
        else:
            good_day = 0
    
        mean_qrs_old = mean_qrs_new
        distances_old = distances_new
    except Exception as e:
        good_day = None; distances_new = None; mean_qrs_new = None
        tb = traceback.extract_tb(e.__traceback__)
        filename, line, _, _ = tb[-1]  # Get the last (innermost) frame
        print(f'DTW_calc failed :{e} (Line: {line}), File: {filename}')
        return good_day, distances_new, mean_qrs_new

    if (str(last_condition) == 'nan') or (last_condition == 'Good'):   
        # Regular case where the last condition was good, so if we get 1 in the DTW calculation, this is as well a good way of wearing          
        DTW_mapping = {0: 'Bad', 1: 'Good'}
    elif last_condition == 'Bad':                     
        # Opposite case, where the last condition was bad, so if we get 1 in the DTW calculation, this is as well a bad way of wearing  
        DTW_mapping = {0: 'Good', 1: 'Bad'}
    
    good_day_series = pd.Series(good_day)        
    good_day_map = good_day_series.map(DTW_mapping)[0]
    
    return good_day_map, distances_new, mean_qrs_new

def merge_excel(old_file, new_file):
    """
    Merges data from two Excel files (represented as dictionaries of DataFrames),
    handling multiple sheets and prioritizing data from the new file.  Handles
    cases where the new file contains sheets not present in the old file.

    For sheets present in both files, a right merge is performed
    based on the 'date' column.  Missing values (NaNs) in the new file's columns
    are filled with corresponding values from the old file, effectively updating
    the data.  If a sheet is only present in the new file, it is copied directly.
    """
    # Initialize an empty dictionary to hold the merged sheets
    merged_sheets = {}

    # Iterate through the sheets (assuming both files have the same sheet names)
    for sheet_name in new_file.keys():  # Focus on all sheets from new_file (right file)
        if sheet_name in old_file:  # Only merge if the sheet exists in both files
            # Perform the right merge based on the 'date' key
            merged_df = pd.merge(old_file[sheet_name], new_file[sheet_name], how='right', on='date', suffixes=('_left', '_right'))

            # Fill missing values in the right file's columns with values from the left file where they exist
            for col in merged_df.columns:
                if col != 'date' and '_left' in col or '_right' in col:  # Check if both left/right exist
                    original_col = col.replace('_left', '').replace('_right', '')
                    left_col = merged_df[original_col + '_left']
                    right_col = merged_df[original_col + '_right']

                    merged_df[original_col] = np.where(right_col.isnull(), left_col, right_col)


            # Drop the temporary columns created by the merge (with '_left' and '_right' suffixes)
            merged_df = merged_df[new_file[sheet_name].columns]

        else:
            # If the sheet is not in old_file, just take the new_file sheet as it is
            merged_df = new_file[sheet_name]

        # Store the merged sheet in the dictionary
        merged_sheets[sheet_name] = merged_df
    return merged_sheets

def export_ecg_data(active_patients, active_patients_SpO2, current, before):
    """
    Export ECG and SpO2 data for active patients to an Excel file, updating or appending new data. 
    The function checks for existing data and processes ECG records to calculate conditions like 
    noise, discontinuity, and heart condition status (good day or bad day) using Dynamic Time Warping (DTW).
    SpO2 data is processed to count the number of sessions based on time gaps.

    :param active_patients: dictionary of active patients with their ECG sensor IDs.
    :param active_patients_SpO2: dictionary of active patients with their SpO2 sensor IDs.
    :param current: current date to process.
    :param before: previous date used as a reference for copying data if current's data doesn't exist.
    :return: updated patient data including the last date, DTW distances, mean QRS, and condition.
    """
    old_excel_filename = base_path + f'/Excel_files/{before}.xlsx'
    new_excel_filename = base_path + f'/Excel_files/{current}.xlsx'

    # Check if the new file already exist, if so load it and later we will combine it with the new data
    if os.path.exists(new_excel_filename):
        # We are loading both the old and new file, to merge them, in case a patient uploaded new data in delay - so in that case we need to merge
        new_file = pd.read_excel(new_excel_filename, sheet_name=None, engine='openpyxl')  # Load all sheets
        old_file = pd.read_excel(old_excel_filename, sheet_name=None, engine='openpyxl')  # Load all sheets
        existing_data = merge_excel(old_file, new_file)
    # Check if the old file already exists, if so load it and later we will combine it with the new data
    elif os.path.exists(old_excel_filename):
        existing_data = pd.read_excel(old_excel_filename, sheet_name=None, engine='openpyxl')
        # Now lets create the new file as a copy of the old data
        shutil.copy(old_excel_filename, new_excel_filename)
    else:
        existing_data = {}  # Initialize an empty dictionary if the file doesn't exist
    
    old_data = pd.read_csv(base_path + '/Excel_files/patients_last_data.csv', index_col='Patient')
    Noise = {}; Discon = {}; Good_day = {}; Distances_new = {}; Mean_qrs_new = {}; Last_date = {}; Percent_data = {}
    base_path_records = base_path + "/Records"
    for patient, sensor in active_patients.items():
        try:
            if patient in old_data.index:
                patient_old_data = old_data.loc[patient]
            else:
                patient_old_data = None
                
            path = os.path.join(base_path_records, patient, 'patch_files', f'ECGRec_{sensor}_{current}_ECG_denoised.csv')  
            if os.path.exists(path): 
                temp = pd.read_csv(path)
                percent = f'{np.round(temp.shape[0] / (24*60*60) * 100, 1)}%'
            else: 
                percent = None
                
            if (patient_old_data is None) or (patient_old_data['Last_date'] < current): 
                # Only if this is a new patient OR the last date we ran QC over this patient, is older than current, we will run the QC
                QRSs_dict = {}; discon = {}; noise = {}; 
                QRSs_dict, noise, discon, good_day, distances_new, mean_qrs_new = None, None, None, None, None, None
                
                if not os.path.exists(path):
                    print(f"The file does not exist: {path}")
                else:    
                    QRSs_dict = get_mean_QRSs(path)
                    good_day, distances_new, mean_qrs_new = DTW_calc(QRSs_dict, patient_old_data)
                    noise, discon = Discontinuity_Noise_check(path)

                    # Only add the patient data if distances_new is not None
                    if distances_new is not None:
                        Distances_new[patient] = distances_new
                        Mean_qrs_new[patient] = mean_qrs_new
                        Last_date[patient] = current
                        Good_day[patient] = good_day
                        Noise[patient] = noise
                        Discon[patient] = discon
                        
            else:    # In case we already ran this date
                current_data = existing_data[patient][existing_data[patient]['date'] == current]
                noise = current_data['Noise'][0]
                discon = current_data['Discon'][0]
                good_day = current_data['Condition'][0]
                
            # SpO2
            num_of_sessions = None
            try:
                spo2_SN = active_patients_SpO2[patient]
                spo2_path = os.path.join(base_path_records, patient, 'patch_files', f'O2_Oxyfit_{spo2_SN}_{current}_O2.csv')
                if os.path.exists(spo2_path):
                    spo2 = pd.read_csv(spo2_path)
                    spo2 = spo2.loc[:, ['Record Time', 'Timezone', 'SpO2']]
                    spo2['time_diff'] = spo2['Record Time'].diff()
                    spo2['session_id'] = (spo2['time_diff'] > 4*60*60*1000).cumsum() # Four hours apart
                    num_of_sessions = spo2['session_id'].max() + 1
                    
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                filename, line, _, _ = tb[-1]  # Get the last (innermost) frame
                print(f'SpO2 problem in {patient}, {spo2_path.split("/")[-1]}: {e} (Line: {line}, File: {filename})')
            

            current_data = pd.DataFrame({
                    'date': [current], 
                    'Noise': [noise],
                    'Discon': [discon],
                    'Condition': [good_day],
                    'ECG Percent': [percent],
                    'SpO2 Sessions (4-hour apart)': [num_of_sessions]
                })        
                
            if patient in existing_data:
                old_patient_data = existing_data[patient]  # Load old sheet data
                if (old_patient_data.iloc[0]['date'] == current_data['date']).all():
                    # in case the date already exist, we will just update the data
                    old_patient_data.iloc[0, old_patient_data.columns.get_loc('Noise')] = current_data['Noise'].values[0]
                    old_patient_data.iloc[0, old_patient_data.columns.get_loc('Discon')] = current_data['Discon'].values[0]
                    old_patient_data.iloc[0, old_patient_data.columns.get_loc('Condition')] = current_data['Condition'].values[0]
                    # In case ECG percent column isn't shown                        
                    if 'ECG Percent' not in old_patient_data.columns:
                        old_patient_data['ECG Percent'] = None
                        old_patient_data[old_patient_data['date'] == current] = percent
                        
                    elif 'ECG Percent' in old_patient_data.columns:
                        old_patient_data.iloc[0, old_patient_data.columns.get_loc('ECG Percent')] = current_data['ECG Percent'].values[0]                            
                    combined_data = old_patient_data                      
                else:
                    combined_data = pd.concat([current_data, old_patient_data], ignore_index=True)  # Add new data on top
            else:
                combined_data = current_data  # No old data, just use the new data
                
            # Write the combined data (new + old) to the new Excel file for current
            if os.path.exists(new_excel_filename):
                with pd.ExcelWriter(new_excel_filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    combined_data.to_excel(writer, sheet_name=patient, index=False)
            else:
                with pd.ExcelWriter(new_excel_filename, engine='openpyxl', mode='w') as writer:
                    combined_data.to_excel(writer, sheet_name=patient, index=False)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)
            filename, line, _, _ = tb[-1]  # Get the last (innermost) frame
            print(f"{patient} ERROR in export_ecg_data: {e} (Line: {line}), File: {filename}")
            continue
                        
    # Delete sheets of inactive patients from the new Excel file
    inactive_patients = set(existing_data.keys()) - set(active_patients.keys())
    if inactive_patients:
        with pd.ExcelWriter(new_excel_filename, engine='openpyxl', mode='a') as writer:
            for patient in inactive_patients:
                writer.book.remove(writer.book[patient])            
    
    for patient in Last_date.keys():
        if Distances_new[patient] is not None:
            old_data.loc[patient, 'Last_date'] = Last_date[patient]
            old_data.loc[patient, 'Distances_new'] = str(Distances_new[patient])
            old_data.loc[patient, 'Mean_qrs_new'] = str(Mean_qrs_new[patient])
            old_data.loc[patient, 'Condition'] = Good_day[patient]
        
    old_data_path = base_path + '/Excel_files/patients_last_data.csv'
    old_data.to_csv(old_data_path)
    
    # Google drive uploads
    upload_file_to_drive(new_excel_filename)

    return old_data

# region main

end_date = datetime.date.today() - timedelta(days=1) # We need to end the analysis yesterday, to make sure all yesterday data was uploaded
start_date = end_date - timedelta(days=14)

# Get active patient info
active_patients_SpO2, active_patients_ECG = get_active_patients()

# Send QC Alerts
if SEND_QC_ALERTS:
    send_exacerbation_alert()
    check_for_stalled_patients(base_path + '/Records',  active_patients_ECG)

# Run the calculations and exports
date_pairs = []
current_date = start_date + timedelta(days=1)
while current_date <= end_date:
    previous_date = current_date - timedelta(days=1)
    date_pairs.append((current_date.strftime('%Y-%m-%d'), previous_date.strftime('%Y-%m-%d')))
    current_date += timedelta(days=1)

for i,couple in enumerate(date_pairs):
    current = couple[0]
    before = couple[1]
    df = export_ecg_data(active_patients_ECG, active_patients_SpO2, current, before)
    print(current, end='\r')

yesterday_date = (datetime.date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
path_excel_file = base_path + f'/Excel_files/{yesterday_date}.xlsx'
if SEND_QC_ALERTS:
    dtw_alerts(path_excel_file)

# Upload old patient data
old_data_path = base_path + '/Excel_files/patients_last_data.csv'
upload_file_to_drive(local_file_path=old_data_path, upload_file_name=f'patients_last_data_{current}.csv')