import os
import glob
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Inputs
subjects_dir = os.environ['SUBJECTS_DIR']
openmp = int(os.environ['FS_OPENMP'])
base_dir = os.environ['BASE_DIR']
n_jobs = int(os.environ['N_JOBS'])
do_rs_connectivity = os.environ['DO_RS_CONNECTIVITY']
generate_report = os.environ['GENERATE_REPORT']
subject = base_dir.split('/')[-1]
epi_subjects_dir = os.path.join(base_dir, 'derivatives', 'anat')
fs_result = os.path.join(epi_subjects_dir, subject, 'mri')
results_dir = os.path.join(base_dir, 'derivatives')
src_spacing = os.environ['SRC_SPACING']
srcfilename = (subject + '_' + src_spacing + '_src.fif')
src_file = os.path.join (results_dir, srcfilename)
bem_file = (epi_subjects_dir + '/' + subject + '/bem/' + subject + '-head.fif')
source_loc_dir = os.path.join(base_dir, 'derivatives', 'source_loc')
epochs_save_dir = os.path.join(source_loc_dir, 'events_and_epochs')
#new_eve_filename = os.path.join(results_dir, eve_file_base + '_processed.csv')

def get_evefiles(dir):
    fn = glob.glob(dir + '/*_Events_processed.csv')
    return fn

def get_rawfiles(dir):
    fn = glob.glob(dir + '/data/*.fif')
    return fn

def strip_path(fn):
    return fn.split('/')[-1]

def get_filebase(f):
    f = f.split('/')[-1]
    f = f.split('.')[0]
    return f

def return_subj_and_block(f):
    f = get_filebase(f)
    f1, f2 = f.split('_')[:2]
    f = str(f1 + '_' + f2)
    return f

def read_pandas_events_csv(e): 
    e = pd.read_csv(e, delimiter=',', float_precision=3, thousands='.', skiprows=0)
    e.reset_index(drop=True, inplace = True)
    e.iloc[:,0].astype(int)
    e.iloc[:,1].astype(str)
    e.iloc[:,2] = e.iloc[:,2].astype(int)
    print(e.dtypes)
    return e

def generate_event_dict (events):
    name_of_events = np.unique(events.iloc[:,1])
    name_of_events = np.sort(name_of_events)
    print(f"\n\n\n\n\n\n\n\n\n\n\n\nEvent-Names for event_dict: {name_of_events}")
    spike_dict=dict()
    spike_dict['ignore_me']=0
    for i in range(name_of_events.size):
        key = (name_of_events[i])
        val = i + 1
        spike_dict[key] = val
    print(f"\n\n\n\nEvent dict is:\n{spike_dict}")
    return spike_dict

def drop_names_from_event_df(e):
    e.iloc[:,1] = 0
    e.values.astype(int)
    return e

def get_event_range():
    rng = len(np.unique(epochs.event_id.items()))
    if rng < 10:
        return rng
    else:
        return 10

raw_files = get_rawfiles(base_dir)
print(f"\n\nThe following raw_files have been detected:")
for f in raw_files:   
    print (f"{strip_path(f)} : {f}")

eve_files = get_evefiles(results_dir)
print(f"\n\nThe following eve_files have been detected:")
for f in eve_files:   
    print (f"{strip_path(f)} : {f}")
print(f"\n\n")
if not os.path.exists(source_loc_dir):
    os.mkdir(source_loc_dir)

os.chdir(source_loc_dir)

for rawfile in raw_files:
    for eve_file in eve_files:
        if return_subj_and_block(eve_file) == return_subj_and_block(rawfile):
            print(f"\n\n--> Now processing rawfile: {rawfile} \n--> with eventfile: {eve_file}")
            raw = mne.io.read_raw_fif(rawfile)  
            noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=180, picks='meg', 
                                                    method='empirical', rank='auto', 
                                                    verbose = True, n_jobs = n_jobs)
            #events = mne.find_events(raw, stim_channel='STI014', shortest_event=0, uint_cast=True)
            events = pd.DataFrame()
            events = read_pandas_events_csv(eve_file)
            event_id = dict()
            event_id = generate_event_dict(events)
            print(f"\n\n\nEvents as loaded via pandas: {events}")
            events = drop_names_from_event_df(events)
            print(f"\n\n\nEvents after name stripping: {events}")
            print(f"\n\n\nEvents.dtypes: {events.dtypes}")
            print(f"Event sample numbers after names were dropped - First: {events.iloc[0,0]}")
            print(f"Event sample numbers after names were dropped - Last: {events.iloc[-1,0]}")
            raw.load_data()
            #events_npy = events.to_numpy(dtype=int)
            #events = mne.read_events(events.values, exclude=0)
            raw.add_events(events, stim_channel='STI014', replace=True)
            filebase = get_filebase(rawfile)
            epochs_filename = os.path.join(results_dir, filebase + '_epo.fif')
            print(epochs_filename)
            if 1==1: #not os.path.isfile(epochs_filename):
                epochs = mne.Epochs(raw, events, event_id, 
                                        tmin=-2, tmax=2, 
                                        baseline=(-2,-1), 
                                        #reject=reject, 
                                        #picks='meg',
                                        min_duration = (4 *raw['sfreq']),
                                        on_missing = 'ignore')
                print(f"\n\n\nEpochs event_ids are: {epochs.event_id}")
                print(f"Events, shape is: {events.shape}") 
                epochs.plot(n_epochs=10)            
                #epochs.save(epochs_filename, overwrite=True)
            else:
                epochs = mne.read_epochs(epochs_filename)

            eventnames = epochs.event_id.keys()
            for eventname in eventnames:
                if str(eventname) == 'ignore_me':
                    pass
                else:
                    plot_event = eventname + '_plot'
                    plot_event = epochs[str(eventname)].load_data().pick('meg').average()
                    filename = (subject + '_' + filebase + '_' + str(plot_event) + '.png')
                    fig = plot_event.plot_topomap(times=np.linspace(-0.25, 0., 5), ch_type='mag', 
                                                time_unit='ms', title=filename)
                    full_filename = os.path.join(source_loc_dir, filename)
                    fig.savefig(full_filename)
    
            #event_range = get_event_range()
            #for k in range(event_range):
            #    print(k)
            #    spikes = dict()
            #    spikes[k] = epochs[k].average().pick('meg')
            #    spikes[k].plot(time_unit='s', noise_cov=noise_cov)
            #    spikes[k].plot_topomap(times=np.linspace(-0.25, 0., 5), ch_type='mag', time_unit='s')

