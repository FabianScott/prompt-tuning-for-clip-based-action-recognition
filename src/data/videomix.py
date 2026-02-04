import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class VideoMixDataset(Dataset):
    def __init__(self, base_dataset, config, max_retries=10):
        self.dataset = base_dataset
        self.mix_prob = config["videomix-prob"]
        self.variant = config["videomix-type"] # spatial | temporal | spatiotemporal
        self.class_names = base_dataset.class_names
        self.samples = base_dataset.samples
        self.num_classes = len(self.class_names)
        self.idx_to_mix = None  # Optional index to always mix with
        self.class_to_idx = base_dataset.class_to_idx
        self.max_retries = max_retries

    def __len__(self):
        return len(self.dataset)

    def set_idx_to_mix(self, idx):
        self.idx_to_mix = idx

    def _sample_mask(self, video_shape):
        T, H, W = video_shape
        M = torch.zeros((T, H, W), dtype=torch.float32)

        if self.variant == "spatial":
            lam = np.random.uniform(0, 1)
            
            wc = np.random.uniform(0, W)
            hc = np.random.uniform(0, H)

            bw = W * np.sqrt(lam)
            bh = H * np.sqrt(lam)

            w1 = int(np.clip(wc - bw / 2, 0, W))
            w2 = int(np.clip(wc + bw / 2, 0, W))
            h1 = int(np.clip(hc - bh / 2, 0, H))
            h2 = int(np.clip(hc + bh / 2, 0, H))

            M = torch.zeros((T, H, W), dtype=torch.float32)
            M[:, h1:h2, w1:w2] = 1.0

        elif self.variant == "temporal":
            t0, t1 = sorted(np.random.randint(0, T, 2))
            M[t0:t1, :, :] = 1.0

        else:  # spatiotemporal
            t0, t1 = sorted(np.random.randint(0, T, 2))
            h0, h1 = sorted(np.random.randint(0, H, 2))
            w0, w1 = sorted(np.random.randint(0, W, 2))
            M[t0:t1, h0:h1, w0:w1] = 1.0

        return M

    def __getitem__(self, idx):
        sample_1 = self.dataset[idx]
        if sample_1 is None:
            return None
        xA, yA = sample_1
        yA = onehot_into_float(yA, num_classes=self.num_classes)

        if np.random.rand() > self.mix_prob:
            return xA, yA

        idxB = np.random.randint(0, len(self.dataset)) if self.idx_to_mix is None else self.idx_to_mix
        other_sample = self.dataset[idxB]
        retries = 0
        while other_sample is None and retries < self.max_retries:
            print(f"Retrying VideoMix sample selection for index {idxB}: attempt {retries}")
            idxB = np.random.randint(0, len(self.dataset))
            other_sample = self.dataset[idxB]
            retries += 1
        if retries == self.max_retries and other_sample is None:
            print(f"Failed to find a valid sample for VideoMix after {self.max_retries} retries. Returning original sample.")
            return xA, yA
        if retries > 0:
            print(f"Successfully found a valid sample for VideoMix after {retries} retries.")
        xB, yB = other_sample
        yB = onehot_into_float(yB, num_classes=self.num_classes)

        V, T, C, H, W = xA.shape
        M = self._sample_mask((T, H, W)).to(xA.device)
        M = M.unsqueeze(0).unsqueeze(2)  # (1, T, 1, H, W)
        M = M.expand(V, -1, -1, -1, -1)  # (V, T, 1, H, W)

        x_mix = M * xA + (1 - M) * xB
        lambda_M = M.mean().item()
        y_mix = lambda_M * yA + (1 - lambda_M) * yB

        return x_mix, y_mix

def onehot_into_float(label, num_classes):
    onehot = torch.zeros(num_classes)
    onehot[label] = 1.0
    return onehot