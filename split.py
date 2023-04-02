# Created by Baole Fang at 4/1/23
import os

import pandas as pd
import torchaudio
import numpy as np


def unseen(root):
    segment=48000
    data=[]
    for spk_id in os.listdir(root):
        for folder in os.listdir(os.path.join(root,spk_id)):
            for file in os.listdir(os.path.join(root,spk_id,folder)):
                wav=os.path.join(spk_id,folder,file)
                length=torchaudio.info(os.path.join(root,wav)).num_frames
                for start in range(0,length-segment,segment):
                    end=start+segment
                    data.append([wav,start,end,spk_id])
    return np.array(data)


if __name__ == '__main__':
    orig_train = pd.read_csv('original/train.csv')
    orig_train = orig_train[orig_train.spk_id.str.startswith('id1')]
    orig_train=orig_train.drop(columns=['ID','duration'])
    positive=orig_train.to_numpy()
    for i in range(len(positive)):
        positive[i,0]=positive[i,0].lstrip('/localscratch/voxceleb1/wav/')
    negative=unseen('/home/baole/cmu/2023spring/11785/data/vox1/vox1_test_wav/wav')
    positive=np.hstack((positive,np.ones((len(positive),1))))
    negative = np.hstack((negative, np.zeros((len(negative), 1))))
    np.random.shuffle(positive)
    np.random.shuffle(negative)
    train=np.vstack((positive[:8000],negative[:8000]))
    val = np.vstack((positive[8000:9000], negative[8000:9000]))
    test=np.vstack((positive[9000:10000], negative[9000:10000]))
    print(test)
    train_df=pd.DataFrame(train,columns=['wav','start','stop','spk_id','label'])
    val_df = pd.DataFrame(val, columns=['wav', 'start', 'stop', 'spk_id','label'])
    test_df=pd.DataFrame(test,columns=['wav','start','stop','spk_id','label'])

    train_df.to_csv('split/speaker/train.csv',index=False)
    val_df.to_csv('split/speaker/val.csv',index=False)
    test_df.to_csv('split/speaker/test.csv',index=False)
