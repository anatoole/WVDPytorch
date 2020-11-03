import filterbank as fb
from filterbank import _extract_image_patches
import soundfile as sf
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.functional as tf
from torch.utils import data
from torch import nn, tensor
import csv

def wvd(signal, window, L, bins, hop, mode, J, Q):

    signal = torch.FloatTensor(signal)
    if len(signal.shape) > 3:
        signal = torch.squeeze(signal)
    if len(signal.shape) > 2:
        signal = torch.squeeze(signal)
    if len(signal.shape) == 1:
        signal = torch.unsqueeze(signal, 0)
    s = torch.stft(signal, n_fft = 2 * window, win_length = window, hop_length = hop)
    s = s[:,:,:,1]+s[:,:,:,0]

    # remodulate the stft prior the spectral correlation for simplicity

    pi = 2*3.1415926
    step = 1 / window
    freq = np.linspace(-step * L, step * L, 2 * L + 1)
    time = np.arange(s.shape[-1]).reshape((-1, 1))
    mask = (np.cos(np.pi * time * freq) + np.sin(np.pi * time * freq)*1j) * np.hanning(2 * L + 1)
    extract_patches = nn.Unfold(kernel_size = (2*L+1,1), stride= (2,1), padding = 2)
    signal = extract_patches(torch.unsqueeze(s, dim = 1))
    s = signal.numpy()
    x = []
    for i in range(s.shape[0]):
        output = np.dot(mask, s[i,:,:] * np.conj(np.flip(s[i,:,:])))
        x.append(output)
    s = np.array(x)
    s = s.astype(np.float)

    filter, mu, cor, sigma, mix = fb.generate_gaussian_filterbank(s.shape[-1], J*Q, s.shape[0], 5, 22050)
    if len(filter.shape) == 4 :
        filter = torch.squeeze(filter)
    if len(filter.shape) == 2 :
        filter = torch.unsqueeze(filter, 0)
#    print(filter.shape)
#    print(s.shape)
    wvd_convolved = torch.bmm(torch.FloatTensor(s), filter)

    return wvd_convolved



def norm(arr):
    return (arr - np.mean(arr) ) / np.std(arr)


class DataSimp(data.Dataset):
    def __init__(self, df):
        super(DataSimp, self)
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self[idx]['sig'], self[idx]['label']


class Dataset(data.Dataset):
    def __init__(self, df, filename=False, white_noise=False, pink_noise=False, rnd_shift=False, mirror=False, gammatone=False):
        super(Dataset, self)
        self.df = df
        self.filename = filename # true if you want the filename in getitem
        self.rng = np.random.RandomState(42)
        self.pink_noise = pink_noise
        self.white_noise = white_noise
        self.rnd_shift = rnd_shift
        self.mirror = mirror
        self.gammatone = gammatone


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df[idx]
        print(row)
        sig = 0.01*np.random.rand(12000)
        fs = 24000


        for root, dirs, files in os.walk("/mnt"):
            for file in files:
                if file == str(row[14]):
                    path = os.path.join(root,file)
                    sig, fs = sf.read(path)
        sig = sig[:,0] if sig.ndim == 2 else sig
        sig = np.concatenate([sig, np.zeros(int(row[12])*fs)])
        sig = sig[int(row[12])*fs:(int(row[12])+5)*fs]
        sig = norm(sig)
        if self.mirror :
            sig = np.flip(sig) if np.random.random() > .5 else sig
        if self.rnd_shift :
            shift = int(np.random.random()*5*fs)
            sig = np.concatenate([sig[shift:], sig[:shift]])

        sig = norm(sig)
        if self.gammatone :
            sig = fftweight.fft_gtgram(sig, fs, 512/fs, 64/fs, 64, 500)

        if self.filename:
            return tensor(sig).unsqueeze(0).float(), float(row[15]), row.path
        else:
            return tensor(sig).unsqueeze(0).float(), float(row[15])


def rewrite(infile):
    edit  = []
    for i in range(len(infile)):
        string = str(infile[i])
        row = string.replace('[','')
        row = row.replace(']','')
        row = row.replace('\"','')
        row = row.replace('\n','')
        row = row.replace(' ','')
        row = row.replace('\'','')
        row = row.replace('?','')
        edit.append(row)
    return(edit)



def array(infile):
    result = []
    with open(infile) as csvfile:
        reader = csv.reader(csvfile) # change contents to floats
        for row in reader: # each row is a list
            result.append(row)
    return(result)

def PrintModel(model, inlength=22050, gammatone=False, indata=None):
    x = tensor(np.arange(inlength)).view(1, 1, -1).float()
    x = fftweight.fft_gtgram(np.arange(inlength), 22050, 512/22050, 64/22050, 64, 500) if gammatone else np.arange(inlength)
    x = tensor(x).float().unsqueeze(0).unsqueeze(0)
    x = indata if indata is not None else x
    print('in shape : ',x.shape, '\n')
    prevshape = x.shape
    for layer in model:
        print(layer)
        x = layer(x)
        if x.shape != prevshape:
            print('Outputs : ',x.shape)
            prevshape = x.shape
        print()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

