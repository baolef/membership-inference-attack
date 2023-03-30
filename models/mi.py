# Created by Baole Fang at 3/29/23

import torch
import torch.nn as nn
from models import register, make, get


@register('mi')
class MI(nn.Module):

    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = get(encoder)
        self.classifier = make(classifier)

    def forward(self, x):
        y = self.encoder.encode_batch(x)
        y = self.classifier.forward(torch.squeeze(y))
        return y

    def train(self, mode=True):
        self.encoder.train(False)
        self.classifier.train(mode)
