########################################### Updated: 10.7.2024 #####################
# - for events the occur for only one day.
# + for events the occur for a period longer than a day

from datetime import datetime
from dateutil import rrule
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

base_path = '/home/ubuntu/pipeline/ForOsherPipeline'

def download_file_from_drive(destination_path, credentials_path=base_path + '/credentials.json'):
    """
    Download a file from Google Drive using its file ID.

    :param file_id: str, the ID of the file in Google Drive
    :param destination_path: str, the local path where the file will be saved
    :param credentials_path: str, path to the service account credentials file
    """
    file_id = '1pwYeZF-vl5s2RQerbX22VP-575xAnEpY' # this is the Devices Tracking file_id
    # Authenticate Google Drive API
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    
    # Request the file
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')

    # Download the file
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download Devices Tracking file from google drive progress: {int(status.progress() * 100)}%")

    fh.close()
download_file_from_drive(base_path + '/Devices_Tracking.xlsx')

# _________________________________________

class Patient:
    def __init__(self, id, start_date=None, end_date=None):
        self.id = id
        print(id)
        pats = get_patients_info()
        pat_info = pats[pats['Patient ID'] == id]
        if start_date is None:                  # No dates were given 
            start_date = pat_info['Trial start'].iloc[0]; end_date = pat_info['Trial end'].iloc[0]; 
            self.start_date = datetime(int(start_date[6:10]), int(start_date[3:5]), int(start_date[:2]))
            if str(pat_info['Trial end'].iloc[0]) == 'nan':     # If the trial didn't end yet
                self.end_date = datetime.today()
            else:
                self.end_date = datetime(int(end_date[6:10]), int(end_date[3:5]), int(end_date[:2]))
        else:                                   # If dates were given in the format %Y-%M-%d
            self.start_date = datetime(int(start_date[:4]), int(start_date[5:7]), int(start_date[8:10]))
            self.end_date = datetime(int(end_date[:4]), int(end_date[5:7]), int(end_date[8:10]))
            
        # Generate list of dates between start_date and end_date
        dates = list(rrule.rrule(rrule.DAILY, dtstart=self.start_date, until=self.end_date))
        # Format dates as strings in 'YYYY-MM-DD' format
        self.dates = [date.strftime('%Y-%m-%d') for date in dates]
        
        self.events = events(self.dates, self.id)
        
        self.sensors = []
        for i in range(len(pat_info['CNS'].iloc[0])):
            if i == 1:
                temp = pat_info['First replace'].iloc[0].split('.')
                F_date = f'{temp[2]}-{temp[1]}-{temp[0]}'
                self.sensors.append((pat_info['CNS'].iloc[0][i],F_date))
            elif i == 2:
                temp = pat_info['Second replace'].iloc[0].split('.')
                Sec_date = f'{temp[2]}-{temp[1]}-{temp[0]}'
                self.sensors.append((pat_info['CNS'].iloc[0][i],Sec_date))
            elif i == 3:
                temp = pat_info['Third replace'].iloc[0].split('.')
                Third_date = f'{temp[2]}-{temp[1]}-{temp[0]}'
                self.sensors.append((pat_info['CNS'].iloc[0][i],Third_date))
            else:
                temp = pat_info['Trial start'].iloc[0].split('-')
                Start_date = f'{temp[2]}-{temp[1]}-{temp[0]}'
                self.sensors.append((pat_info['CNS'].iloc[0][i], Start_date))
        
        self.SpO2 = pat_info['SpO2 SN'].iloc[0]

    def __str__(self):
        return f"Patient {self.id}: Start Date: {self.start_date}, Events: {self.events}"
    

# _________________________________________
def get_patients_info():
    # We need the Devices Tracking file location
    path = base_path + '/Devices_Tracking.xlsx'
    # path = r"G:\.shortcut-targets-by-id\18nJLFf-lQ4hga_I6F0O8D_z8FGWBVA_W\Clinical\Devices Tracking\Devices Tracking.xlsx"

    pats = pd.read_excel(path)
    pats = pats[(~pats['Trial start'].isna()) & (~pats.Site.isna()) & (pats['Patient ID'] != 'Unused Kit')]
    pats['Trial start'] = pats['Trial start'].str.replace('.', '-', regex=False)
    pats['1st Visit'] = pats['1st Visit'].str.replace('.', '-', regex=False)
    pats['Trial end'] = pats['Trial end'].str.replace('.', '-',  regex=False)
    ecg_cols = [col for col in pats.columns if "ECG" in col]
    spo2_cols = [col for col in pats.columns if "SpO2" in col]

    pats.loc[pats['Patient ID'].isin(['DB03009', 'EA03010', 'AD04002', 'AA04003','GM02001']), 'Trial end'] = 'out'

    cnss = []; SpO2SN = []
    for index, row in pats[:].iterrows():
        cns = [string[:14].replace('/','_') for string in row[ecg_cols].to_list() if str(string) != 'nan']
        spo2s = [str(string)[:10] for string in row[spo2_cols].to_list() if str(string) != 'nan']
        cnss.append(cns)
        SpO2SN.append(spo2s)

    pats['CNS'] = cnss; pats['SpO2 SN'] = SpO2SN
    pats['SpO2 SN'] = pats['SpO2 SN']
    pats = pats[['Patient ID', 'Trial start', '1st Visit', 'Trial end', 'CNS', 'Site', 'SpO2 SN', 'First replace', 'Second replace', 'Third replace']]
    return pats

    
# _________________________________________
def events(dates, id):
    # Some symbols I have used:
    # In the beginning: + marks an event which occured on consecutive days. - markd a single day event.
    # +MIDDLE_ marks a case where another event is occuring while there is another event which occured on consecutive days, 
    # for example AF04006 had hospitalization during 27.5-11.6 and had dialisys on the 29.5.
    # the end letter: M for major event, m for minor event.
    
    events = {}
    for date in dates:
        events[date] = None
    
    # ___________________________Ichilov____________________
    if id == 'DK03001':
        events['2023-12-03'] = '-Breath Shortness \nLow SpO2_m';
        events['2023-12-29'] = '-Coldness_m'; 
        events['2024-01-30'] = '-Exacerbation \n Hospitalization_M'
        events['2024-02-28'] = '-Hard Breathing_m'; 
        events['2024-03-12'] = '+Hard Breathing \n Low SpO2 \n Cough_m'
        for i in range(dates.index('2024-03-13'),dates.index('2024-03-18')+1):
            events[dates[i]] = '+_m'
        events['2024-03-21'] = '-Improvement_m'
        events['2024-04-03'] = '-Improvement_m'
        events['2024-04-11'] = '-Improvement_m'
    elif id == 'AY03002':
        events['2024-02-12'] = '-Hard breathing_m'
        events['2024-03-14'] = '-Low SpO2_m'
        events['2024-04-03'] = '-Past weeks: \nhard to walk_m'
        events['2024-07-02'] = '-Low SpO2_m'
        events['2024-07-08'] = '-Low SpO2_m'
        events['2024-07-15'] = '-Low SpO2_m'
        events['2024-07-22'] = '-Weakness Low SpO2_m'
        events['2024-07-23'] = '+Hospitalization_M'
        for i in range(dates.index('2024-07-24'),dates.index('2024-07-29')+1):
            events[dates[i]] = '+_M'
    # elif id == 'FC03004':       # No events
    #     None
    # elif id == 'MC03003':       # No events
    #     None
    elif id == 'CH03005':       # No events
        events['2024-10-26'] = '+Hard Breathing \nhigh fever \nsnot \n_m'
        for i in range(dates.index('2024-10-27'),dates.index('2024-10-29')+1):
            events[dates[i]] = '+_m'
            
    elif id == 'SD03008':
        events['2024-11-27'] = '+Exacerbation_M'
        for i in range(dates.index('2024-11-28'),dates.index('2024-12-01')+1):
            events[dates[i]] = '+_M'
        
    # ___________________________Soroka____________________
    elif id == 'AF04006':
        events['2024-05-05'] = '-Breath Shortness_m';
        events['2024-05-06'] = '+Hospitalization, \nBreath Shortness_M'; 
        events['2024-05-07'] = '+_M'
        events['2024-05-09'] = '+Hospitalization, \nventilation_M'; 
        for i in range(dates.index('2024-05-10'),dates.index('2024-05-16')+1):
            if dates[i] == '2024-05-11':
                events[dates[i]] = '+MIDDLE_Stop \nventilation_M'
            else:
                events[dates[i]] = '+_M'
        events['2024-05-27'] = '+Hospitalization, \nExacerbation, \nCatheterization_M'
        for i in range(dates.index('2024-05-28'),dates.index('2024-06-11')+1):
            if dates[i] == '2024-06-03':
                events[dates[i]] = '+MIDDLE_Dialysis_M'
            else:
                events[dates[i]] = '+_M'
            
    elif id == 'MA04007':
        events['2024-05-09'] = '-Chest pains'
        events['2024-06-17'] = '+Hard breath, weakness'
        for i in range(dates.index('2024-06-18'),dates.index('2024-06-24')+1):
            events[dates[i]] = '+'
        events['2024-06-17'] = '-Small Improvement'
    elif id == 'ID04001':
        events['2024-03-07'] = '-General regression'
        events['2024-03-25'] = '-Last weeks:\nGeneral regression, \nHard breath'
        events['2024-04-03'] = '-More regress in last weeks:\nGeneral regression, \nHard breath '
    elif id == 'DP04005':
        events['2024-03-28'] = '-Tiredness in \nthe past week_m'
        events['2024-04-17'] = '-Cough, coldness_m'
    elif id == 'EA04004':
        events['2024-02-18'] = '-Hospitalization \nHard breathing_m'
        events['2024-02-28'] = '-Suffocation feeling_m'
        events['2024-03-07'] = '-Past week: \nHigh HR \nin easy effort_m'
        events['2024-03-14'] = '-Last days: coughing_m'
        events['2024-03-15'] = '-Hospitalization \nBreath Shortness \nChest pain_m'
        events['2024-04-02'] = '-High HR, \npalpitations_m'
        events['2024-04-09'] = '+Coldness, fever, sore throat_m'
        events['2024-04-10'] = '+_m'
        events['2024-05-16'] = '-Hospitalization \nHigh HR \nArrythmia_m'

    elif id == 'RS04009':
        events['2024-08-13'] = '-After Bronchoscopy \nLow SpO2_m'
        events['2024-09-23'] = '+Hospitalization \nHard Breathing \nLow SpO2_M'
        for i in range(dates.index('2024-09-24'),dates.index('2024-09-26')+1):
            events[dates[i]] = '+_M'
        events['2024-10-22'] = '-Occasionally Low SpO2_m'
        events['2024-11-05'] = '-Antibiotics, oxygen_m'
        # events['2024-12-03'] = '-Weakness, \nIncreased phlegm'

    # ___________________________Assuta____________________
    elif id == 'SA02003':
        events['2024-06-10'] = '+Corona - Fever\nHeadace, Cough_m'
        events['2024-06-11'] = '+_m'
        events['2024-09-18'] = '-Dry cough \n chest hurts_m'
        events['2024-10-15'] = '-Low SpO2 \nhard breathing_m'
        events['2024-10-30'] = '+COPD Exacerbation_M'
        for i in range(dates.index('2024-10-31'),dates.index('2024-11-03')+1):
            events[dates[i]] = '+_M'
    # elif id == 'ZB02002':     # No events
    #     None
    # elif id == 'GM02001':     # No events
    #     None

    return events


# _________________________________________
def get_DatesSensors(patient):    
    data = pd.read_excel(base_path + '/Devices_Tracking.xlsx', sheet_name=1)
    patient_info = data[data['Patient ID'] == patient]
    info = {}
    info['SN0'] = [(patient_info['ECG Patch SN'].iloc[0][:14])]
    info['SN0'].append(patient_info['Trial start'].iloc[0].replace('.','/'))
    
    if not pd.isna(patient_info['ECG Patch SN.1'].iloc[0]): 
        info['SN1'] = [patient_info['ECG Patch SN.1'].iloc[0][:14]]
        info['SN1'].append(patient_info['First replace'].iloc[0].replace('.','/'))
        
    if not pd.isna(patient_info['ECG Patch SN.2'].iloc[0]): 
        info['SN2'] = [patient_info['ECG Patch SN.2'].iloc[0][:14]]
        info['SN2'].append(patient_info['Second replace'].iloc[0].replace('.','/'))
    
    if not pd.isna(patient_info['ECG Patch SN.3'].iloc[0]): 
        info['SN3'] = [patient_info['ECG Patch SN.3'].iloc[0][:14]]
        info['SN3'].append(patient_info['Third replace'].iloc[0].replace('.','/'))
    
    if pd.isna(patient_info['Trial end'].iloc[0]):
        info['end date'] = datetime.now().strftime('%Y/%m/%d')
    else:
        info['end date'] = patient_info['Trial end'].iloc[0].replace('.','/')
    return info
     



