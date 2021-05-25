import torch
import numpy as np

class NoiseModule:
    def __init__(self, num_imgs=1, num_maps=1, img_size=(320,320), ALPHA=0.01, device=None):
        super(NoiseModule, self).__init__()
        self.num_imgs = num_imgs
        self.num_maps = num_maps
        self.h, self.w = img_size
        self.num_pixels = self.h * self.w
        self.alpha = ALPHA
        self.device = device
        self.noise_variance = torch.ones(self.num_imgs * self.num_pixels * 2)
        self.emp_var = torch.zeros(self.num_imgs * self.num_pixels * 2)

    def get_index(self, arr=None, img_idx=None):
        if arr is None:
            arr = self.noise_variance
        idx = img_idx * self.num_pixels * 2
        return arr[idx:idx+self.num_pixels * 2], np.arange(idx, idx+self.num_pixels * 2)

    def get_index_multiple(self, arr=None, img_idxs=None):
        if arr is None:
            arr = self.noise_variance
        noise = np.zeros((len(img_idxs), self.num_pixels*2), dtype=np.float)
        noise_idx = np.zeros((len(img_idxs), self.num_pixels*2), dtype=np.float)
        for key, img_idx in enumerate(img_idxs):
            idx = img_idx * self.num_pixels
            noise[key] = arr[idx:idx+self.num_pixels * 2]
            noise_idx[key] = np.arange(idx, idx+self.num_pixels * 2)
        return noise, noise_idx

    def loss_fast(self, var1, var2):
        noise_loss = 0
        for idx in range(var1.shape[0]):
            covar1 = var1[idx].to(self.device) + 1e-6
            covar2 = var2[idx].to(self.device) + 1e-6
            ratio = 1. * (covar1 / covar2)
            loss = -0.5 * (torch.log(ratio) - ratio + 1).to(self.device)
            loss = abs(loss)
            noise_loss += torch.sum(loss) / var1.shape[1]
        return noise_loss / var1.shape[0]

    def sample_noise(self, idxs):
        samples = torch.zeros(len(idxs), 2, self.h, self.w)
        for idx, img_idx in enumerate(idxs):
            var, var_idx = self.get_index(self.noise_variance, img_idx)
            var = var.view(2, self.h, self.w).to(self.device)
            samples[idx] = var
        return samples

    def update(self):
        print("Updating Noise")
        print("----------------------------------------")
        self.noise_variance = self.noise_variance + self.alpha * (self.emp_var - self.noise_variance)
        print(f'Max: {torch.max(self.noise_variance)}, Min :{torch.min(self.noise_variance)}')
        print("----------------------------------------")

    def save(self, path):
        noise_array = self.noise_variance.numpy()
        print('saving the noise')
        np.save(path,noise_array)

    def load(self, path):
        noise_array = np.load(path)
        self.noise_variance = torch.Tensor(noise_array).float()