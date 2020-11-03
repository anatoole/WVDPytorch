#from filterbank import STFT
import torch
from torch import nn
import utils as u

get = {
'AlexNet' : nn.Sequential(

    nn.Conv2d(1, kernel_size = 11, stride = 4, out_channels = 96),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 3, stride = 2),

    nn.Conv2d(96, padding = 2, kernel_size = 5, groups = 2, out_channels = 256),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 3, stride = 2),

    nn.Conv2d(256, padding = 1, kernel_size = 3, out_channels = 384),
    nn.ReLU(),

    nn.Conv2d(384, padding = 1, kernel_size = 3, groups = 2, out_channels = 384),
    nn.ReLU(),

    nn.Conv2d(384, padding = 1, kernel_size = 3, groups = 2, out_channels = 256, stride = 1),
    nn.ReLU(),
    #nn.MaxPool2d(kernel_size = 3, stride = (6,2)),
    nn.AdaptiveMaxPool2d(output_size=(1,1)),
    u.Flatten(),
    nn.Linear(256, 1024),
    nn.ReLU(),
    nn.Dropout(p = 0.5),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(p = 0.5),
    nn.Linear(1024, 208),
    nn.Softmax(),
),

'model.0' : nn.Sequential(

    nn.Conv1d(227, padding = (0,1), kernel_size = (1,3), stride = 1, out_channels = 100),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,3), stride = 2),

    nn.Conv1d(100, padding = (0,1), kernel_size = (1,3), out_channels = 64, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),

    nn.Conv1d(64, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),

    nn.Conv1d(128, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),

    nn.Conv1d(128, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),

    nn.Conv1d(128, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),

    nn.Linear(3, out_features = 1024),
    nn.Dropout(p = 0.5),
    nn.Linear(1024, out_features = 512),
    nn.Dropout(p = 0.5),
    nn.Linear(512,2),
    nn.Softmax(-1),
),

'model.1' : nn.Sequential(

    nn.Conv2d(1, padding = (0,1), kernel_size = (1,3), stride = 1, out_channels = 100),
    nn.ReLU(),
    nn.MaxPool2d(padding = (0,0), kernel_size = (1,3), stride = 2),

    nn.Conv2d(100, padding = (0,1), kernel_size = (1,3), out_channels = 64, stride = 1),
    nn.ReLU(),
    nn.MaxPool2d(padding = (0,0), kernel_size = (1,2), stride = 2),

    nn.Conv2d(64, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool2d(padding = (0,0), kernel_size = (1,2), stride = 2),

    nn.Conv2d(128, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool2d(padding = (0,0), kernel_size = (1,2), stride = 2),

    nn.Conv2d(128, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool2d(padding = (0,0), kernel_size = (1,2), stride = 2),

    nn.Conv2d(128, padding = (0,1), kernel_size = (4,3), out_channels = 1, stride = (3,1)),
    nn.ReLU(),
    nn.MaxPool2d(padding = (0,0), kernel_size = (1,2), stride = 2),
#    nn.MaxPool2d(kernel_size = (1,2), stride = 2),
    nn.Linear(3, out_features = 1024),
    nn.Dropout(p = 0.5),
    nn.Linear(1024, out_features = 512),
    nn.Dropout(p = 0.5),
    nn.Linear(512,10),
    nn.Softmax(dim = -1),
),


'Wignerville' : nn.Sequential(
#    STFT(int(0.046439909 * 22050 + .5), 315),
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), stride = 1, out_channels = 100),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,3), stride = 2),
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), out_channels = 64),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),
    nn.Linear(1, out_features = 1024),
    nn.Dropout(p = 0.5),
    nn.Linear(1, out_features = 512),
    nn.Dropout(p = 0.5),
    nn.Linear(1,2),
    nn.Softmax(),
),





'model_raw' : nn.Sequential(
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), stride = 1, out_channels = 100),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,3), stride = 2),
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), out_channels = 64),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),
    nn.Conv1d(1, padding = (0,1), kernel_size = (1,3), out_channels = 128, stride = 1),
    nn.ReLU(),
    nn.MaxPool1d(padding = (0,0), kernel_size = (1,2), stride = 2),
    nn.Linear(1, out_features = 1024),
    nn.Dropout(p = 0.5),
    nn.Linear(1, out_features = 512),
    nn.Dropout(p = 0.5),
    nn.Linear(1,2),
    nn.Softmax(),
),




'invention' : nn.Sequential(
        torch.nn.Conv1d(in_channels = 501, out_channels = 1, padding = 1, kernel_size = 64, stride = (2,)),
        torch.nn.BatchNorm1d(1),
        torch.nn.ReLU(),
        #torch.nn.Conv1d(8, 1, kernel_size = 3, stride = 1)
        torch.nn.AdaptiveMaxPool1d(1),
)
}


