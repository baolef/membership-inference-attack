# Created by Baole Fang at 2/20/23

import torch
import os
import torchaudio
import csv

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_root):

        self.audios = []
        self.labels = []
        self.ids = []

        with open(root) as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                path=os.path.join(data_root, row['wav'])
                num_frames=int(row['stop'])-int(row['start'])
                start=int(row['start'])
                audio, _ = torchaudio.load(path, num_frames=num_frames, frame_offset=start)
                self.audios.append(audio.flatten())

                self.labels.append([float(row['label'])])
                self.ids.append(row['spk_id'])

        self.length = len(self.audios)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        audio = self.audios[ind]
        label = torch.tensor(self.labels[ind])
        spk_id = self.ids[ind]
        return audio, label, spk_id


def create_dataloader(root, data_root, batch_size):
    train_data = AudioDataset(os.path.join(root, 'train.csv'), data_root)
    val_data = AudioDataset(os.path.join(root, 'val.csv'), data_root)
    test_data = AudioDataset(os.path.join(root, 'test.csv'), data_root)

    n_cpu = os.cpu_count()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        num_workers=n_cpu,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=val_data,
        num_workers=n_cpu // 2,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        num_workers=n_cpu // 2,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader
