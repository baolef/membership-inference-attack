# Created by Baole Fang at 4/25/23

from speechbrain.pretrained import EncoderClassifier
from data import create_dataloader
import torch
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from speechbrain.lobes.augment import _prepare_openrir
from hyperpyyaml import load_hyperpyyaml


def augmentation(x,hparams):
    wavs = x
    lens = torch.tensor([x.shape[1]]*x.shape[0]).cuda()
    wavs_aug_tot = []
    wavs_aug_tot.append(wavs)
    for count, augment in enumerate(hparams['augment_pipeline']):

        # Apply augment
        wavs_aug = augment(wavs, lens)

        # Managing speed change
        if wavs_aug.shape[1] > wavs.shape[1]:
            wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
        else:
            zero_sig = torch.zeros_like(wavs)
            zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
            wavs_aug = zero_sig

        if hparams['concat_augment']:
            wavs_aug_tot.append(wavs_aug)
        else:
            wavs = wavs_aug
            wavs_aug_tot[0] = wavs

    wavs = torch.cat(wavs_aug_tot, dim=0)
    n_augment = len(wavs_aug_tot)
    lens = torch.cat([lens] * n_augment)
    return wavs, lens


def compute_features(model, loader, hparams):
    features=[]
    labels=[]
    for i, data in enumerate(loader):
        x, label, y = data
        x = x.cuda()
        aug, _ = augmentation(x, hparams)
        with torch.inference_mode():
            embedding = model.encode_batch(aug).cpu().squeeze(1)
            distances=cosine_similarity(embedding,embedding)
            distances=distances[~np.eye(distances.shape[0], dtype=bool)].reshape(distances.shape[0], -1) # delete diagonal
        features.append([distances.mean(),distances.std()])
        labels.append(label)
    return torch.tensor(features).numpy(), torch.tensor(labels).numpy()


def pipeline(model_name,dir,root):
    _prepare_openrir('openrir', 'openrir/reverb.csv', 'openrir/noise.csv', 3)
    with open("augmentation.yaml") as f:
        hparams = load_hyperpyyaml(f, None)
    if os.path.exists(os.path.join(root, 'train.npy')):
        train = np.load(os.path.join(root, 'train.npy'))
        val = np.load(os.path.join(root, 'val.npy'))
        test = np.load(os.path.join(root, 'test.npy'))
    else:
        if not os.path.exists(root):
            os.makedirs(root)
        model = EncoderClassifier.from_hparams(source="speechbrain/{}".format(model_name), run_opts={"device": "cuda"})
        model.eval()
        train_loader, val_loader, test_loader = create_dataloader(
            root=os.path.join('split', dir),
            data_root='/home/baole/cmu/2023spring/11785/data/vox1/vox1_all/wav',
            batch_size=1
        )

        train_x, train_y = compute_features(model, train_loader, hparams)
        val_x, val_y = compute_features(model, val_loader, hparams)
        test_x, test_y = compute_features(model, test_loader, hparams)

        train = np.column_stack((train_x, train_y))
        np.save(os.path.join(root, 'train.npy'), train)
        val = np.column_stack((val_x, val_y))
        np.save(os.path.join(root, 'val.npy'), val)
        test = np.column_stack((test_x, test_y))
        np.save(os.path.join(root, 'test.npy'), test)
    return train,val,test

