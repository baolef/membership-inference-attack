# Created by Baole Fang at 2/20/23

import torch
import os
import torchaudio


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self):
        audio, _ = torchaudio.load('speechbrain/tests/samples/single-mic/example1.wav')
        self.audio = audio.flatten()
        self.label = torch.tensor([1.0])

    def __len__(self):
        return 100

    def __getitem__(self, ind):
        return self.audio, self.label


def create_dataloader(root, batch_size):
    train_data = AudioDataset()
    val_data = AudioDataset()
    test_data = AudioDataset()

    n_cpu = os.cpu_count()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        num_workers=n_cpu,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        num_workers=n_cpu // 2,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        num_workers=n_cpu // 2,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False
    )
    return train_loader, valid_loader, test_loader
