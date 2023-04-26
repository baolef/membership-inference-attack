# Created by Baole Fang at 4/26/23

from speechbrain.pretrained import EncoderClassifier
from data import create_dataloader
import torch
import numpy as np
import os


def compute_features(model, loader):
    features=[]
    labels=[]
    for i, data in enumerate(loader):
        x, label, y = data
        x = x.cuda()
        with torch.inference_mode():
            pred, score, _, _ = model.classify_batch(x)
        features.append([score])
        labels.append(label)
    return torch.tensor(features).numpy(), torch.tensor(labels).numpy()


def pipeline(model_name,dir,root):
    if os.path.exists(os.path.join(root, 'train.npy')):
        train = np.load(os.path.join(root, 'train.npy'))
        val = np.load(os.path.join(root, 'val.npy'))
        test = np.load(os.path.join(root, 'test.npy'))
    else:
        if not os.path.exists(root):
            os.makedirs(root)
        model = EncoderClassifier.from_hparams(source="speechbrain/"+model_name, run_opts={"device": "cuda"})
        model.eval()
        train_loader, val_loader, test_loader = create_dataloader(
            root=os.path.join('split', dir),
            data_root='/home/baole/cmu/2023spring/11785/data/vox1/vox1_all/wav/',
            batch_size=1
        )

        train_x, train_y = compute_features(model, train_loader)
        val_x, val_y = compute_features(model, val_loader)
        test_x, test_y = compute_features(model, test_loader)

        train = np.column_stack((train_x, train_y))
        np.save(os.path.join(root, 'train.npy'), train)
        val = np.column_stack((val_x, val_y))
        np.save(os.path.join(root, 'val.npy'), val)
        test = np.column_stack((test_x, test_y))
        np.save(os.path.join(root, 'test.npy'), test)
    return train,val,test
