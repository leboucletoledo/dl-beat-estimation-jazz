
import math
import os
import numpy as np
import pandas as pd
from librosa import load, get_duration
from madmom.audio.signal import Signal
from madmom.processors import SequentialProcessor
from madmom.utils import quantize_events
from scipy.ndimage import maximum_filter1d
from torch import manual_seed
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from tqdm.auto import tqdm

# Use a negative integer to avoid confusing with non-beat positions in arrays
MASK_VALUE = -1 

class BeatDataset(Dataset):
    def __init__(self, 
                 dir_annotations: str, 
                 dir_metadata: str, 
                 dir_audio: str, 
                 pre_processor: SequentialProcessor,
                 name: str=None,
                 resample_rate=None,
                 pad_frames: int=None,
                 mono_processing: bool=True,
                 trim_lead: float=None,
                 trim_end: float=None
                ):
        self.name = name
        self.pre_proc = pre_processor
        self.rs_rate = resample_rate
        self.pad_frames = pad_frames
        self.mono = mono_processing
        self.trim_lead = trim_lead
        self.trim_end = trim_end

        # Load annotations and audio information
        self.df_ann = pd.read_pickle(dir_annotations, compression='bz2')
        self.df_info = pd.read_pickle(dir_metadata, compression='bz2')
        self.dir_audio = dir_audio
        self.data = {idx: {} for idx, _ in self.df_info.iterrows()}

        for idx, _ in tqdm(self.data.items(), desc='Sample'):
            # General info for each sample
            melid = self.df_info['melid'][idx]
            self.data[idx]['melid'] = melid
            title = self.df_info['title'][idx]
            self.data[idx]['title'] = title

            # Obtain beat positions
            series = self.df_ann.loc[self.df_ann['melid'] == melid]
            beat_onsets = series['onset'].to_numpy(dtype=float)
            beat_nums = series['beat'].to_numpy(dtype=int)
            
            # Obtain audio and sampling rate
            audio, sr = self.get_audio(idx)

            if self.trim_lead is not None and self.trim_lead!=0:
                # Also eliminate beat onsets
                trim_ids = [beat_onsets > self.trim_lead][0]
                beat_onsets = beat_onsets[trim_ids]
                beat_onsets -= self.trim_lead
                # Also match indeces for beat numbering array
                beat_nums = beat_nums[trim_ids]
            if self.trim_end is not None and self.trim_end!=0:
                # Also eliminate beat onsets
                audio_length = get_duration(y=audio, sr=sr)
                trim_ids = [beat_onsets < audio_length][0]
                beat_onsets = beat_onsets[trim_ids]
                # Also match indeces for beat numbering array
                beat_nums = beat_nums[trim_ids]
            
            # Need to provide the audio file and sample rate to generate madmom signal
            signal = Signal(audio, sample_rate=sr)
            
            # Run the provided pre-processor (and use a different resample rate than 44100)
            if self.rs_rate is None:
                self.rs_rate = 44100
        
            x = self.pre_proc(signal, rs_rate=self.rs_rate)
            self.data[idx]['signal'] = x

            # Store beat onsets and obtain downbeat onsets (only for datasets that have labelled beats)
            self.data[idx]['beats_onsets'] = beat_onsets
            if beat_nums.any(): 
                downbeats = beat_nums == 1
                downbeat_onsets = beat_onsets[downbeats]
                self.data[idx]['downbeats_onsets'] = downbeat_onsets
            else:
                self.data[idx]['downbeats_onsets'] = None
            
            # Quantize onsets with madmom utilies
            self.data[idx]['beats_quantized'] = quantize_events(beat_onsets, fps=self.pre_proc.fps, length=len(x))
            try:
                self.data[idx]['downbeats_quantized'] = quantize_events(downbeat_onsets, fps=self.pre_proc.fps, length=len(x))
            except:
                print('No downbeat information for track, masking with value {}'.format(MASK_VALUE))
                self.data[idx]['downbeats_quantized'] = np.ones(len(x), dtype='float32') * MASK_VALUE          

    def __len__(self):
        return len(list(self.data.keys()))

    def __getitem__(self, idx):
        x = self.data[idx]['signal']
        bt = self.data[idx]['beats_quantized']
        dnbt = self.data[idx]['downbeats_quantized']

        if self.pad_frames is not None:
            x = self._add_padding(x, self.pad_frames)
        
        # Extra dimension to match output from PyTorch model
        x = np.expand_dims(x, axis=0)
        bt = np.expand_dims(bt, axis=0)
        dnbt = np.expand_dims(dnbt, axis=0) 
        
        return x, bt, dnbt, idx
        
    def _add_padding(self, data, pad_frames):
        pad_start = np.repeat(data[:1], pad_frames, axis=0)
        pad_stop = np.repeat(data[-1:], pad_frames, axis=0)
        return np.concatenate((pad_start, data, pad_stop))

    def get_audio(self, idx):
        title = self.data[idx]['title']
        sample_path = os.path.join(self.dir_audio, title + '.wav')
        # better resampling than torchaudio, no tensors and avoids overloading GPU
        audio, sr = load(sample_path, mono=self.mono) 

        if self.trim_lead is not None and self.trim_lead!=0:
            lead_frames = math.floor(sr * self.trim_lead)
            audio = audio[lead_frames:]
        if self.trim_end is not None and self.trim_lead!=0:
            end_frames = math.floor(sr * self.trim_end)
            audio = audio[:-end_frames]

        return audio, sr
    
    def split_data(self, val_size, test_size=None, seed=None):
        lengths = []
        total_length = len(list(self.data.keys()))
        print('Number of total items in dataset: ', total_length)
        if test_size is None:
            train_size = 1. - val_size 
        else:
            train_size = 1. - (val_size + test_size)
            lengths.append(math.floor(total_length * test_size + 0.5))
        
        assert train_size>=0.50, 'Training set must be at least 0.50 of the total length'
        
        if seed is not None:
            seed = manual_seed(seed)
        try:
            lengths.insert(0, math.floor(total_length * val_size))
            lengths.insert(0, math.floor(total_length * train_size + 0.5))
            sets = random_split(self, lengths, generator=seed)
        except ValueError:
            lengths[1] = math.floor(total_length * val_size + 0.5)
            sets = random_split(self, lengths, generator=seed)
        finally:
            print('sum of lengths:', sum(lengths))
            print('dataset length', total_length)
            return sets         

    def widen_targets(self, size=3, value=0.50):
        for idx in self.data.values():
            np.maximum(idx['beats_quantized'], maximum_filter1d(idx['beats_quantized'], size=size) * value, out=idx['beats_quantized'])
            # skip masked targets without downbeat information
            if np.allclose(idx['downbeats_quantized'], MASK_VALUE):
                continue
            else:
                np.maximum(idx['downbeats_quantized'], maximum_filter1d(idx['downbeats_quantized'], size=size) * value, out=idx['downbeats_quantized'])
        print('Targets width extended')

def load_dataset(dataset: Dataset, 
                 val_size: float=0.2, 
                 mini_set: bool=False, 
                 mini_set_ratio: int=5,
                 num_workers: int=0,
                 pin_memory: bool=False,
                 ):
    # Split into needed subsets
    data_train, data_val = dataset.split_data(val_size=val_size)

    # For a mini-version of sets for testing purposes
    if mini_set:
        print('Mini-set enabled')
        data_train = Subset(data_train, list(range(0, int(len(data_train)/mini_set_ratio))))
        data_val = Subset(data_val, list(range(0, int(len(data_val)/mini_set_ratio))))
        try:
            data_test = Subset(data_test, list(range(0, int(len(data_test)/mini_set_ratio))))
            print('Tracks in test set: ', len(data_test))
        except NameError:
            print('No test data found')
    print('Tracks in train set: ', len(data_train))
    print('Tracks in val set: ', len(data_val))

    # Use batch_size=1 instead of batch_size=None, for matching tensor shape
    train_loader = DataLoader(data_train, batch_size=1, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory
                            )
    val_loader = DataLoader(data_val, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory
                            )
    
    return train_loader, val_loader