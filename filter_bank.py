from torch import nn
import torch
import numpy as np



def freq_to_mel(f, option="linear"):
    # convert frequency to mel with
    if option == "linear":

        # linear part slope
        f_sp = 200.0 / 3

        # Fill in the log-scale part
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = min_log_hz / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region
        mel = min_log_mel + np.log(f / min_log_hz) / logstep
        return np.where(f >= min_log_hz, mel, f / f_sp)
    else:
        return 2595 * np.log10(1 + f / 700)

def get_scaled_freqs(f0, f1, J):
    f0 = np.array(freq_to_mel(f0))
    f1 = np.array(freq_to_mel(f1))
    m = np.linspace(f0, f1, J)
    # first we compute the mel scale center frequencies for
    # initialization
    f_sp = 200.0 / 3

    # And now the nonlinear scale
    # beginning of log region (Hz)
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp   # same (Mels)

    # step size for log region
    logstep = np.log(6.4) / 27.0

    # If we have vector data, vectorize
    freqs = min_log_hz * np.exp(logstep * (m - min_log_mel))
    freqs /= freqs.max()
    return torch.FloatTensor(freqs)


def generate_gaussian_filterbank(N, M, J, f0, f1, modes=1):
    freqs = get_scaled_freqs(f0, f1, J)
    freqs *= (J-1)*10

    mu = torch.stack([freqs, 0.1 * torch.randn(J * modes)], 1)
    cor = 0.01 * torch.randn(J * modes)
    sigma = torch.stack([freqs/6, 1.+ 0.01 * torch.randn(J * modes)], 1)
    mixing = torch.ones((modes, 1, 1))
    coeff = torch.prod(torch.sqrt(torch.abs(sigma)+0.1),1)*0.95
    Id = torch.eye(2)
    cov = Id * torch.unsqueeze((torch.abs(sigma)+0.1),1) + torch.flip(Id,[0]) * (torch.tanh(cor) * coeff).reshape((-1,1,1))
    cov_inv = torch.inverse(cov)


    # get the gaussian filters
    time = torch.linspace(-5, 5, M)
    freq = torch.linspace(0, J * 10, N)
    x, y = torch.meshgrid(time, freq)
    grid = torch.stack([torch.flatten(y), torch.flatten(x)], 1)
    centered = grid - torch.unsqueeze(mu, 1)

    gaussian = torch.exp(-(torch.matmul(centered, cov_inv)**2).sum(-1))
    norm = torch.norm(gaussian, 2, 1, keepdim=True)
    gaussian_2d = torch.abs(mixing) * torch.reshape(gaussian / norm, (J, modes, N, M))
    return gaussian_2d.sum(1, keepdims=True), mu, cor, sigma, mixing

def _extract_signal_patches(signal, window_length, hop=1, data_format="NCW"):
    if hasattr(window_length, "shape"):
        assert window_length.shape == ()
    else:
        assert not hasattr(window_length, "__len__")

    if data_format == "NCW":
        if signal.ndim == 2:
            signal_3d = signal[:, None, :]
        elif signal.ndim == 1:
            signal_3d = signal[None, None, :]
        else:
            signal_3d = signal

        N = (signal_3d.shape[2] - window_length) // hop + 1
        indices = np.arange(window_length) + np.expand_dims(np.arange(N) * hop, 1)
        indices = np.reshape(indices, [1, 1, N * window_length])
        patches = np.take_along_axis(signal_3d, indices, 2)
        output = np.reshape(patches, signal_3d.shape[:2] + (N, window_length))
        if signal.ndim == 1:
            return output[0, 0]
        elif signal.ndim == 2:
            return output[:, 0, :]
        else:
            return output
    else:
        error


def _extract_image_patches(
    image, window_shape, hop, data_format="NCHW", mode="valid"
):
    p1 = window_shape[0] - 1
    p2 = window_shape[1] - 1
#    print(image.shape)
    image = np.pad(
        image,
        [(0, 0), (0, 0), (p1 // 2, p1 - p1 // 2), (p2 // 2, p2 - p2 // 2)],
    )
#    hop = (hop, hop)
    if data_format == "NCHW":
#        print(image.shape)
        # compute the number of windows in both dimensions
        N = (
            (image.shape[2] - window_shape[0]) // hop[0] + 1,
            (image.shape[3] - window_shape[1]) // hop[1] + 1,
        )

        # compute the base indices of a 2d patch
        patch = np.arange(np.prod(window_shape)).reshape(window_shape)
        offset = np.expand_dims(np.arange(window_shape[0]), 1)
        patch_indices = patch + offset * (image.shape[3] - window_shape[1])

        # create all the shifted versions of it
        ver_shifts = np.reshape(
            np.arange(N[0]) * hop[0] * image.shape[3], (-1, 1, 1, 1)
        )
        hor_shifts = np.reshape(np.arange(N[1]) * hop[1], (-1, 1, 1))
        all_cols = patch_indices + np.reshape(np.arange(N[1]) * hop[1], (-1, 1, 1))
        indices = patch_indices + ver_shifts + hor_shifts

        # now extract shape (1, 1, H'W'a'b')
        flat_indices = np.reshape(indices, [1, 1, -1])
        # shape is now (N, C, W*H)
        flat_image = np.reshape(image, (image.shape[0], image.shape[1], -1))
        # shape is now (N, C)
        patches = np.take_along_axis(flat_image, flat_indices, 2)
        return np.reshape(patches, image.shape[:2] + N + tuple(window_shape))


class STFT(nn.Module):
    def __init__(self, winsize, hopsize, complex=False):
        super(STFT, self).__init__()
        self.winsize = winsize
        self.hopsize = hopsize
        self.register_buffer('window',
                             torch.hann_window(winsize, periodic=False))
        self.complex = complex

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        result = super(STFT, self).state_dict(destination, prefix, keep_vars)
        # remove all buffers; we use them as cached constants
        for k in self._buffers:
            del result[prefix + k]
        return result

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # ignore stored buffers for backwards compatibility
        for k in self._buffers:
            state_dict.pop(prefix + k, None)
        # temporarily hide the buffers; we do not want to restore them
        buffers = self._buffers
        self._buffers = {}
        result = super(STFT, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._buffers = buffers
        return result

    def forward(self, x):
        print(x.shape)
        # we want each channel to be treated separately, so we mash
        # up the channels and batch size and split them up afterwards
        batchsize, channels = x.shape[:2]
        x = x.reshape((-1,) + x.shape[2:])
        # we apply the STFT
#        x = u.wvd(x, window = self.window, hop = self.hopsize, bins = self.winsize, L=6, mode = 'valid')
        if not self.complex:
            x = x.norm(p=2, dim=-1)
        # Apply filter
        filter, mu, cor, sigma, mix = generate_gaussian_filterbank(wvd.shape[-1], 64, wvd.shape[0], 8, 22050)
        wvd_convolved = torch.bmm(wvd, torch.squeeze(filter))
        # restore original batchsize and channels in case we mashed them
        x = x.reshape((batchsize, channels, -1) + x.shape[2:])
        return x

def stft(signal, window, hop, apod, nfft=None, mode="valid"):

    assert signal.ndim == 3
    if nfft is None:
        nfft = window
    if mode == "same":
        left = (window + 1) // 2
        psignal = np.pad(signal, [[0, 0], [0, 0], [left, window + 1 - left]])
    elif mode == "full":
        left = (window + 1) // 2
        psignal = np.pad(signal, [[0, 0], [0, 0], [window - 1, window - 1]])
    else:
        psignal = signal

    apodization = apod(window).reshape((1, 1, -1))

    p = _extract_signal_patches(psignal, window, hop) * apodization
    assert nfft >= window
    pp = np.pad(p, [[0, 0], [0, 0], [0, 0], [0, nfft - window]])
    S = np.fft.fft(pp)
    return S[..., : int(np.ceil(nfft / 2))].transpose([0, 1, 3, 2])
