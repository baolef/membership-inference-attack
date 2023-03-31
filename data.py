# Created by Baole Fang at 2/20/23

import torch
import os
import torchaudio
import csv
from speechbrain.dataio.batch import PaddedBatch

class AudioDataset(torch.utils.data.Dataset):
    def __init__(root, data_root):

        self.audioPath = []
        self.labels = []

        with open(root) as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                del row['']
                self.audioPath.append(data_root + row['wav'])

                if row['present_label'] == "TRUE":
                    self.labels.append(1.0)
                else:
                    self.labels.append(0.0)


        self.length = len(self.audioPath)
        

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        audio, _ = torchaudio.load(self.audioPath[ind])
        audio = audio.flatten()

        label = torch.tensor(self.labels[ind])

        return audio, label


def create_dataloader(root, data_root, batch_size):

    train_data = AudioDataset(root+'/train.csv', data_root)
    val_data = AudioDataset(root+'/val.csv', data_root)
    test_data = AudioDataset(root+'/test.csv', data_root)

    n_cpu = os.cpu_count()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        num_workers=n_cpu,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        collate_fn = PaddedBatch
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        num_workers=n_cpu // 2,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn = PaddedBatch
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        num_workers=n_cpu // 2,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        collate_fn = PaddedBatch
    )

    return train_loader, valid_loader, test_loader
