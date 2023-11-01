import os
import random
import numpy as np
from librosa import clicks
from IPython.display import Audio, display
from scipy.io import wavfile

from funcs_misc import is_notebook

def play_audio(waveform, sample_rate: int) -> None:
  assert is_notebook(), 'Function only intended for Jupyter notebook'
  if waveform.ndim == 1:
    display(Audio(waveform, rate=sample_rate))
  elif waveform.ndim == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

def get_audio_clicks(dataset, 
                     pred_type: str='downbeats_predictions',
                     idx: int=None, 
                     seed: int=None, 
                     x_lims: list=[], 
                     verbose: bool=False,
                     ):
    
    def match_click_size(audio: np.array, 
                         clicks: np.array, 
                         verbose: bool
                         ):
        if audio.shape[0]>clicks.shape[0]:
            diff = audio.shape[0] - clicks.shape[0]
            clicks = np.concatenate([clicks, np.zeros((diff))])
        elif audio.shape[0]<clicks.shape[0]:
            diff = clicks.shape[0] - audio.shape[0]
            clicks = clicks[:-diff]      
        if verbose:
            print('Audio shape: ', audio.shape)
            print('Clicks shape: ', clicks.shape)     
        return clicks
    
    assert pred_type in ['downbeats_predictions',
                         'bars_predictions']
    if idx is None:
        if seed:
            random.seed(seed)
        idx = random.randint(0, len(dataset))
        print('Dataset random index: ', idx)
    else:
        assert idx in list(range(0, len(dataset))), 'No such index for dataset'
    
    audio, sr = dataset.get_audio(idx)
    dnbt_preds = dataset.data[idx][pred_type]
    bt_indices = np.argwhere(dnbt_preds[:,1]!=1)[:,0]
    dnbt_indices = np.argwhere(dnbt_preds[:,1]==1)[:,0]
    
    bt_preds = dnbt_preds[bt_indices][:,0]#.astype('float32')
    dnbt_preds = dnbt_preds[dnbt_indices][:,0]#.astype('float32')

    bt_clicks = clicks(times=bt_preds, 
                       sr=22050, 
                       hop_length=512, 
                       click_freq=1600.0, 
                       click_duration=0.1,
                       )
    dnbt_clicks = clicks(times=dnbt_preds, 
                         sr=22050, 
                         hop_length=512, 
                         click_freq=800.0, 
                         click_duration=0.1,
                         )

    bt_clicks = match_click_size(audio, 
                                     bt_clicks, 
                                     verbose=verbose,
                                     )
    dnbt_clicks = match_click_size(audio, 
                                       dnbt_clicks, 
                                       verbose=verbose,
                                       )

    audio_clicks = audio + bt_clicks + dnbt_clicks

    if x_lims:
        assert len(x_lims)==2, 'x_lims must be of length 2'
        audio_clicks = audio_clicks[int(x_lims[0] * sr):int(x_lims[1] * sr)]

    return audio_clicks, sr

def export_audio_clicks(audio_clicks: np.ndarray, 
                        sr: int, 
                        name: str, 
                        name_suffix: str='',
                        output_dir: str='./output/',
                        norm_16int: bool=True,
                        ) -> None:
    
    def normalize_float_to_16int(waveform: np.ndarray):
        waveform_norm = 2 * (waveform - np.amin(waveform)) / (np.amax(waveform) - np.amin(waveform)) - 1
        waveform_norm *= 32767
        waveform_norm = waveform_norm.astype(np.int16)
        return waveform_norm

    os.makedirs(output_dir, exist_ok=True)
    if norm_16int and (audio_clicks.dtype=='float64' or audio_clicks.dtype=='float32'):
        print('Converting to 16-bit audio file')
        audio_clicks = normalize_float_to_16int(audio_clicks)
        name = name + name_suffix + '_16int.wav'
    elif audio_clicks.dtype=='float64':
        audio_clicks = audio_clicks.astype('float32')
    else:
        name = name + name_suffix + '.wav'

    # Export audio files
    wavfile.write(output_dir + name, rate=sr, data=audio_clicks.T)