import filter_bank as fb
from filter_bank import _extract_image_patches
from filter_bank import stft
import soundfile as sf
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.functional as tf
from torch.utils import data
from torch import nn, tensor
import csv
import matplotlib.pyplot as plt

def wvd(signal, window, L, bins, hop, J, Q, device = 'cuda'):

    #signal = torch.FloatTensor(signal)
#    if len(signal.shape) > 3:
#        signal = torch.squeeze(signal)
#    if len(signal.shape) > 2:
#        signal = torch.squeeze(signal)
#    if len(signal.shape) == 1:
#        signal = torch.unsqueeze(signal, 0)

    s = stft(signal, window, hop, apod = np.hanning, nfft = 2*window, mode = 'valid')
#    s = plt.specgram(signal, NFFT = window, noverlap = window - hop, Fs = 24000)
# remodulate the stft prior the spectral correlation for simplicity
#    print(s.shape)
#    tffw = s
    pi = 2*3.1415926
    step = 1 / window
    freq = np.linspace(-step * L, step * L, 2 * L + 1)
    time = np.arange(s.shape[-1]).reshape((-1, 1))
    mask = (np.cos(pi * time * freq) + np.sin(pi * time * freq)*1j) * np.hanning(
        2 * L + 1
    )
    #extract_patches = nn.Unfold(kernel_size = (2*L+1,1), stride= (2,1), padding = 2)
    patches = _extract_image_patches(s, (2*L+1,1), (2,1))[...,0]
    #s = signal.numpy()
    #x = []
    #for i in range(s.shape[0]):
    #output = np.dot(mask, s[i,:,:] * np.conj(np.flip(s[i,:,:])))
#    print(type(patches))
    output = (patches * np.conj(np.flip(patches, -1)) * mask).sum(-1).real
    x = output
    #x.append(output)
    #s = np.array(x)
#    s = output.astype(np.float)
#    tffw = np.log(np.abs(output))

#    filter, mu, cor, sigma, mix = fb.generate_gaussian_filterbank(bins, 64, J*Q, 5, 24000)

#    filter1, mu, cor, sigma, mix = fb.generate_gaussian_filterbank(s.shape[0], 64, s.shape[-1], 5, 24000)
#    print(filter1.shape, x.shape)
#    if l&en(filter.shape) == 4 :
#        filter = torch.squeeze(filter)
#    if len(filter.shape) == 2 :
#        filter = torch.unsqueeze(filter, 0)
#    s = torch.squeeze(torch.Tensor(s), dim=1)
#    wvd_2 = F.conv2d(torch.Tensor(tffw), torch.Tensor(filter1))[:,:,0]
 #   print(filter.shape, output.shape) #, s.shape)

#    wvd_convolved_log = F.conv1d(torch.Tensor(output).to(device), torch.Tensor(filter).to(device))[:,:,0]

#    wvd_bmm_log = torch.bmm(torch.Tensor(-np.log(np.abs(output))).to(device), torch.Tensor(filter1).to(device))[:,:,0]
#    wvd_convolved = F.conv2d(torch.Tensor(output).to(device), torch.Tensor(filter).to(device))[:,:,0]
#    wvd_bmm = torch.bmm(torch.Tensor(output).to(device), torch.Tensor(filter1).to(device))[:,:,0]

 #   output = F.conv2d(torch.Tensor(output).to(device), torch.Tensor(filter).to(device))
 #   print(wvd_convolved.shape)
#wvd_convolved = torch.bmm(s, filter)
#    print(wvd_convolved.shape)
#    return  wvd_convolved_log.cpu(), wvd_bmm_log.cpu(), wvd_convolved.cpu(), wvd_bmm.cpu()
    return output


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

