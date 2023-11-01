import numpy as np
from scipy.ndimage import maximum_filter1d
from madmom.processors import SequentialProcessor
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, DBNBarTrackingProcessor

# Standard values for audio processing
FPS = 100
FFT_SIZE = 2048
NUM_BANDS = 12

class PreProcessor(SequentialProcessor):
    def __init__(self, rs_rate=44100, frame_size=FFT_SIZE, num_bands=NUM_BANDS, log=np.log, add=1e-6, fps=FPS):
        # Resample to a fixed sample rate in order to get always the same number of filter bins
        sig = SignalProcessor(num_channels=1, sample_rate=rs_rate)
        # Split audio signal in overlapping frames
        frames = FramedSignalProcessor(frame_size=frame_size, fps=fps)
        # Compute STFT
        stft = ShortTimeFourierTransformProcessor()
        # Filter the magnitudes
        filt = FilteredSpectrogramProcessor(num_bands=num_bands)
        # Scale them logarithmically
        spec = LogarithmicSpectrogramProcessor(log=log, add=add)
        # Instantiate a SequentialProcessor
        super(PreProcessor, self).__init__((sig, frames, stft, filt, spec, np.array))
        # Save the fps as attribute (needed for quantization of events)
        self.fps = fps
        
        
class DBNPostProcessor(SequentialProcessor):
    def __init__(self, 
                 min_bpm, 
                 max_bpm, 
                 fps, 
                 beats_per_bar=None, 
                 transition_lambda=100, 
                 threshold=0.05, 
                 meter_change_prob=1e-3, 
                 observation_weight=4,
                 ):
        if beats_per_bar is None:
            beats_per_bar = [3, 4]
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.fps = fps
        self.bpb = beats_per_bar
        self.t_lambda = transition_lambda
        self.thrsh = threshold
        self.mcprob = meter_change_prob
        self.obs_weight = observation_weight

        # Define trackers
        self.bt_tracker = DBNBeatTrackingProcessor(min_bpm=self.min_bpm, max_bpm=self.max_bpm, fps=self.fps, transition_lambda=self.t_lambda, threshold=self.thrsh)
        self.dnbt_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=self.bpb, min_bpm=self.min_bpm, max_bpm=self.max_bpm, fps=self.fps, transition_lambda=self.t_lambda)
        self.bar_tracker = DBNBarTrackingProcessor(beats_per_bar=self.bpb, meter_change_prob=meter_change_prob, observation_weight=observation_weight)

    def track_beats(self, beat_activations):
        return self.bt_tracker(beat_activations)
                                        
    def track_downbeats(self, beat_activations, downbeat_activations):
        combined_activations = np.vstack((np.maximum(beat_activations - downbeat_activations, 0), downbeat_activations)).T
        return self.dnbt_tracker(combined_activations)

    def track_bars_joint(self, beat_activations, downbeat_activations):
        beats = self.track_beats(beat_activations)
        beat_idx = (beats * self.fps).astype(int)
        bar_act = maximum_filter1d(downbeat_activations, size=3)
        bar_act = bar_act[beat_idx]
        bar_act = np.vstack((beats, bar_act)).T
        try:
            bars = self.bar_tracker(bar_act)
        except IndexError:
            bars = np.empty((0, 2))
        return bars