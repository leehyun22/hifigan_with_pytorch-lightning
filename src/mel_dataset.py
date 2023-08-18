import torch
import torchaudio
from torchaudio.transforms import Resample
import random
from torch.utils.data import Dataset
from utils import load_filepaths_and_text
import layers
from typing import List
from omegaconf import DictConfig
from audio_processing import mel_spectrogram


class MelDataset(Dataset):
    """
    1) load audio, text pairs
    2) normalize text and convert embedded vector
    3) compute mel-spec from audio
    4) return mel_spec, audio, audio_path, mel_spec for mel_loss
    """
    def __init__(self, audio_text_path: str, hparams: DictConfig):
        self.audiopaths_text = load_filepaths_and_text(audio_text_path)
        self.data_path = hparams.data_path
        self.is_resample = hparams.is_resample
        self.segment_size = hparams.segment_size
        self.sampling_rate = hparams.sampling_rate
        self.split = hparams.split
        self.fine_tuning = hparams.fine_tuning

        self.hparams = hparams        
        self.n_fft = self.hparams.melkwargs.n_fft
        self.num_mels = self.hparams.melkwargs.n_mels
        self.hop_size = self.hparams.melkwargs.hop_length
        self.win_size = self.hparams.melkwargs.win_length
        self.fmin = self.hparams.melkwargs.f_min
        self.fmax = self.hparams.melkwargs.f_max
        self.fmax_loss = self.hparams.melkwargs.fmax_for_loss

        if self.is_resample:
            self.resample = Resample(
                orig_freq=hparams.ori_sampling_rate,
                new_freq=hparams.sampling_rate
            )

    def get_item_hifi_gan(self, audiopath_text: List):
        audiopath = audiopath_text[0]
        audiopath = self.data_path + '/' + audiopath
        audio, sampling_rate = torchaudio.load(audiopath)
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)
        if self.is_resample:
            audio = self.resample(audio)
        else:
            if sampling_rate != self.sampling_rate:
                raise ValueError(f"{sampling_rate} {self.sampling_rate} SR doesn't match target SR")
        
        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start+self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
        
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = mel_spectrogram(
            y=audio,
            n_fft=self.n_fft,
            num_mels=self.num_mels,
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_size,
            win_size=self.win_size,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        # TODO: 1.fine-tuning process, 2.split case
        mel_spec = torch.squeeze(melspec, 0)
        
        mel_loss = mel_spectrogram(
            y=audio,
            n_fft=self.n_fft,
            num_mels=self.num_mels,
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_size,
            win_size=self.win_size,
            fmin=self.fmin,
            fmax=self.fmax_loss,
        )
        
        return (mel_spec.squeeze(), audio.squeeze(0),audiopath, mel_loss.squeeze())

    def get_mel(self, audiopath: str):
        # default: Normalized
        audio, sampling_rate = torchaudio.load(audiopath)
        if audio.size(0) !=1:
            audio = torch.mean(audio, dim=0).unsqueeze(0)
        if self.is_resample:
            audio = self.resample(audio)
        else:
            if sampling_rate != self.sampling_rate:
                raise ValueError(f"{sampling_rate} {self.sampling_rate} SR doesn't match target SR")
        
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        return self.get_item_hifi_gan(self.audiopaths_text[index])

    def __len__(self):
        return len(self.audiopaths_text)
