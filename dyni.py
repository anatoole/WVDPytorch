import h5py
from models import get
import time
import filterbank as fb
import utils as u
import numpy as np
import argparse
import torch
from torch import nn, tensor, utils, device, cuda, optim, long, save
from torch.utils import data
import dataload
from utils import rewrite
from utils import array
import matplotlib.pyplot as plt
import subprocess


#ARGUMENTS

parse = argparse.ArgumentParser()
parse.add_argument('--option', type=str, choices=['quebec', 'bird', 'AudioMNIST', 'MNIST_raw'])
parse.add_argument('-J', type=int, default=8)
parse.add_argument('-Q', type=int, default=8)
parse.add_argument('--bins', type=int, default=512)
parse.add_argument('-BS', type=int, default=8)
parse.add_argument('-L', type=int, default=6)
parse.add_argument('-LR', type=float, default=0.1)
parse.add_argument('--hop', type=int, default=0)
parse.add_argument('-momentum', type=float, default = 0.9)
parse.add_argument('--epochs', type=int, default=100)
parse.add_argument('-wdl', type=float, default = 0.002)
args = parse.parse_args()


if args.hop == 0:
    args.hop = args.bins // 2


#CREATE MODEL

modelname = 'model.mdl'
model = get['AlexNet']
print('nb param', sum(m.numel() for m in model.parameters() if m.requires_grad))
bad_batches = {}
cuda0 = device('cuda:1')
model.to(cuda0)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std = 0.1)
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std = 0.005)
    torch.nn.init.constant_(m.bias, val=0)


#FETCH DATA

optimizer = optim.SGD(model.parameters(), weight_decay = args.wdl, lr=args.LR, momentum = args.momentum)
print(optimizer)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : .9**epoch)

if args.option == 'bird':
    wavs_train, wavs_test, wavs_valid, label_train, label_test, labels_valid  = dataload.loadbird(args.bins*2, args.L, args.bins, args.hop, args.J, args.Q, mode = 'autre')
    wavs_train = torch.FloatTensor(wavs_train)
    labels = torch.unsqueeze(torch.FloatTensor(label_train), dim=1)
    train = [{'data': wav, 'label' : lab} for wav, lab in zip(wavs_train, labels)]
    loader1 = utils.data.DataLoader(train, batch_size = args.BS, shuffle = True, num_workers = 2, pin_memory = True)


if args.option == 'AudioMNIST':
    wavs_train, labels_train, wavs_test, labels_test, wavs_valid, labels_valid  = dataload.loadmnist()
    wavs_train = torch.squeeze(torch.FloatTensor(wavs_train))
    print(wavs_test.shape)
    labels = torch.unsqueeze(torch.FloatTensor(labels_train), dim=1)
    train = [{'data': wav, 'label' : lab} for wav, lab in zip(wavs_train, labels)]
    loader1 = utils.data.DataLoader(train, batch_size = args.BS, shuffle = True, num_workers = 2, pin_memory = True)


if args.option == 'MNIST_raw':

    train, labels, wavs_test, labels_test, wavs_valid, labels_valid, record = dataload.loadraw(args.bins*2, args.L, args.bins, args.hop, args.J, args.Q, mode = 'autre')
    train = torch.squeeze(torch.squeeze(torch.Tensor(train)))
    labels = torch.unsqueeze(torch.Tensor(labels), -1)
    train = [{'data': wav, 'label' : lab} for wav, lab in zip(train, labels)]
    loader1 = utils.data.DataLoader(train, batch_size = args.BS, shuffle = True, num_workers = 2, pin_memory = True)

if args.option == 'quebec':
    wavs_train, labels_train, wavs_test, labels_test  = dataload.load_quebec()
    labels_train = torch.unsqueeze(torch.Tensor(labels_train), dim=1)
    labels_test = torch.unsqueeze(torch.FloatTensor(labels_test), 1)
    test = [{'data': wav, 'label' : lab} for wav, lab in zip(wavs_test, labels_test)]
    train = [{'data': wav, 'label' : lab} for wav, lab in zip(wavs_train, labels_train)]
    loader1 = utils.data.DataLoader(train, batch_size = args.BS, shuffle = True, num_workers = 2, pin_memory = True)
print(len(wavs_train), len(labels_train))


loss_fun = nn.CrossEntropyLoss()

loss, tprs, tnrs, accs = [], [], [], []
print('Started at ',time.ctime(time.time()))


#TRAINING LOOP


for epoch in range(args.epochs):

    loss, tprs, tnrs, accs = [], [], [], []
    model.train()
    scheduler.step()
    print(float(epoch/args.epochs))

    for batch in loader1 :

        x = torch.FloatTensor(np.asarray([h5py.File(batch['data'][i][:-1], 'r')['data'][...] for i in range(len(batch['label']))]))
        label = batch['label']

        x = torch.unsqueeze(x, dim=1)
        x = x.cuda(cuda0)
        label = label.cuda(cuda0)
        pred = model(x)
        label = torch.squeeze(label, dim = 1)
        if len(label.shape) == 2:
            label = torch.squeeze(label, dim=1)
        score = loss_fun(pred, label.long())
        pred = torch.argmax(pred, dim=1)

        label = label.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        torch.autograd.set_detect_anomaly(True)
        label = label.astype(int)
        pred = pred.astype(int)
        accs.append(np.sum(label==pred)/len(label))
        tprs.append((label&pred).sum()/label.sum())
        tnrs.append((~label&~pred).sum()/(~label).sum())


        score.backward()
        optimizer.step()

        loss.append(score.item())

    save(model.state_dict(), modelname[:-3]+'stdc')
    with open('loss', 'a') as f:
        f.write("%s\n" % loss)
    with open('accs', 'a') as f:
        f.write("%s\n" % accs)
    with open('tprs', 'a') as f:
        f.write('%s\n' % tprs)
    with open('tnrs', 'a') as f:
        f.write('%s\n' % tnrs)


# TEST MODEL
    with torch.no_grad():
        print('boucle valid')
        model.eval()
        labels, preds, losses, fullpreds = [], [], [], []

        scheduler.step()

        for batch in utils.data.DataLoader(test, batch_size = int(args.BS/2), shuffle=True, num_workers=2, pin_memory=True):
            x = torch.FloatTensor(np.asarray([h5py.File(batch['data'][i][:-1], 'r')['data'][...] for i in range(len(batch['label']))]))
            labels = batch['label']
            labels = labels.cuda(cuda0)
            x = torch.unsqueeze(x, dim=1)
            x = x.cuda(cuda0)
            pred = model(x)

            labels = torch.squeeze(labels, dim=1)
            if len(labels.shape) == 2:
                labels = torch.squeeze(labels, dim = 1)

            score = loss_fun(pred, labels.long())
            pred = torch.argmax(pred, dim=1)
            labels = labels.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            losses.append(score)

    print('suite')

    with open('roc', 'a') as f:
       f.write("%s\n" % ['labels=',labels, 'preds=',pred])


    labels = np.array(labels).astype(int)
    preds = (np.array(pred)>0).astype(int)
    validacc = np.sum(labels==preds) / len(labels)
    validtpr = np.sum(labels&preds) / np.sum(labels)
    validtnr = np.sum(~labels&~preds) / np.sum(~labels)

    with open('valid', 'a') as g:
        g.write('%s\n' % validacc)
    with open('losses', 'a') as g:
        g.write('%s\n' % torch.mean(torch.tensor(losses)))
    with open('validtpr', 'a') as g:
        g.write('%s\n' % validtpr)
    with open('vlaidtnr', 'a') as g:
        g.write('%s\n' % validtnr)
