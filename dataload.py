import soundfile as sf
import annotation as an
import os
import h5py
from tqdm import tqdm
import numpy as np
import time
from wvdutils import wvd
from torch import device

cuda0 = device('cuda:0')


def loadraw(window, L, bins, hop, J, Q, mode):

        # Loading the file
        path = "/nfs/NAS4/data_anatole/AudioMNIST/preprocessed_data/"
        names = ["Audio"]
        types = ["digit_0", "gender_1", "gender_2", "gender_3", "gender_0", "digit_1", "digit_2", "digit_3", "digit_4"]
        sets = ["test", "train", "validate"]
        basefile = "{}Net_{}_{}.txt"
        wavs_test = list()
        wavs_train = list()
        label_train = list()
        label_test = list()
        wavs_valid = list()
        label_valid = list()

        t0 = time.time()

        for name in names:
            if name == 'Alex':
                chiffre = 0
            else :
                chiffre = 1
            for type in types :
                for set in sets :
                    text = open(path + basefile.format(name, type, set), 'r')
                    for line in tqdm(
                        text, ascii = True, desc = "recording {}".format(chiffre)
                    ):
                        f = h5py.File(str(line)[:-1], "r")

                        for filename in f:
                            if set == 'train':
                                if filename == 'data':
                                    wavs_train.append(wvd(f[filename][...],window, L, bins, hop, mode, J, Q))
                                if filename == 'label':
                                    label_train.append(int(line[8]))
                            if set == 'validate':
                                if filename == 'data':
                                    wavs_valid.append(wvd(f[filename][...], window, L, bins, hop, mode, J, Q))
                                if filename == 'label':
                                    label_valid.append(int(line[8]))
                            if set == 'test':
                                if filename == 'data':
                                    wavs_test.append(wvd(f[filename][...], window, L, bins, hop, mode, J, Q))
                                if filename == 'label':
                                    label_test.append(int(line[8]))


        wavs_train = np.array(wavs_train) #.astype("float32")
        label_train = np.array(label_train) #.astype("int32")
        wavs_valid = np.array(wavs_valid) #.astype("float32")
        label_valid = np.array(label_valid) #.astype("int32")
        wavs_test = np.array(wavs_test) #.astype("float32")
        label_test = np.array(label_test) #.astype("int32")


        print("Dataset loaded in {0:.2f}s.".format(time.time() - t0))

        return wavs_train, label_train, wavs_test, label_test, wavs_valid, label_valid


def loadbird(window, L, bins, hop, J, Q, mode):

    path = '/nfs/NAS4/data_anatole/Birdvox/'
    basefile = 'BirdVox-70k_unit{}.hdf5'
    names = ['01'] #, '02', '03', '05', '07', '10']
    glob = list()
    lab = list()
    for name in names:
        f = h5py.File(path + basefile.format(name), 'r')
        for k in tqdm((f['waveforms'].keys())):
            audio = np.array(wvd(f['waveforms'][k],window, L, bins, hop, mode, J, Q))
            label = int(str(f['waveforms'][k])[38])
            glob.append(audio)
            lab.append(label)
    a = int(len(glob)/2)
    b = int(len(glob)*3/4)
    glob = np.array(glob)
    lab = np.array(lab)
    wavs_train = glob[:a,:]
    wavs_test = glob[a:b,:]
    wavs_valid = glob[b:,:]
    label_train = lab[:a]
    label_test = lab[a:b]
    label_valid = lab[b:]
    return wavs_train, wavs_test, wavs_valid, label_train, label_test, label_valid

def loadmnist():
        # Loading the file
        path = "/nfs/NAS4/data_anatole/AudioMNIST/preprocessed_data/"
        names = ["Audio"]
#        types = ["gender_0", "gender_1", "gender_2", "gender_3", "digit_0", "digit_1", "digit_2", "digit_3", "digit_4"]
        types = ["digit_0"] #, "digit_1", "digit_2", "digit_3", "digit_4"]
        sets = ["train"] #, "test"] #, "validate"]
        basefile = "{}Net_{}_{}.txt"
        wavs_test = list()
        wavs_train = list()
        label_train = list()
        label_test = list()
        wavs_valid = list()
        label_valid = list()

        t0 = time.time()

        for name in names:
            if name == 'Alex':
                chiffre = 0
            else :
                chiffre = 1
            for type in types :
                for set in sets :
                    text = open(path + basefile.format(name, type, set), 'r')
                    for line in tqdm(
                        text, ascii = True, desc = "recording {}".format(chiffre)
                    ):
                         f = h5py.File(str(line)[:-1], "r")
                         for filename in f:
                             if set == 'train':
                                 if filename == 'data':
                                      wavs_train.append(f[filename][...])
                                 if filename == 'label':
                                     label_train.append(int(line[8]))
                             if set == 'validate':
                                 if filename == 'data':
                                     wavs_valid.append(f[filename][...])
                                 if filename == 'label':
                                     label_valid.append(int(line[8]))
                             if set == 'test':
                                 if filename == 'data':
                                      wavs_test.append(f[filename][...])
                                 if filename == 'label':
                                     label_test.append(int(line[8]))
        label_test = np.array(label_test)
        wavs_train = np.array(wavs_train)
        print(wavs_train.shape)
        wavs_test = np.array(wavs_test)
        label_valid = np.array(label_valid)
        wavs_valid = np.array(wavs_valid)
        label_train = np.array(label_train)
        print("Dataset loaded in {0:.2f}s.".format(time.time() - t0))

        return wavs_train, label_train, wavs_test, label_test, wavs_valid, label_valid

def load_quebec():

        # Loading the file
        path = "/nfs/NAS4/data_anatole/Quebec/300/"
        sets = ["train", "test"]
        basefile = "{}set"
        wavs_test = list()
        wavs_train = list()
        label_train = list()
        label_test = list()

        t0 = time.time()

        for set in sets :
            text = open(path + basefile.format(set), 'r')
            for row in tqdm(
                text, ascii = True, desc = "Loading data"
            ):
                f = h5py.File(str(row)[:-1], "r")
                for filename in f:
                     if set == 'train':
                         if filename == 'data':
                             wavs_train.append(row)
                         if filename == 'label':
                             label_train.append(len(f[filename][...])-1)
                     if set == 'test':
                         if filename == 'data':
                             wavs_test.append(row)
                         if filename == 'label':
                             label_test.append(len(f[filename][...])-1)

        label_test = np.array(label_test)
        wavs_train = np.array(wavs_train)
        s = len(wavs_train)%8
        wavs_test = np.array(wavs_test)
        label_train = np.array(label_train)
#        wavs_train = wavs_train[:8+s]
#        label_train = label_train[:8+s]
#        wavs_test = wavs_test[:8+s]
#        label_test = label_test[:8+s]

        print("Dataset loaded in {0:.2f}s.".format(time.time() - t0))

        return wavs_train, label_train, wavs_test, label_test

