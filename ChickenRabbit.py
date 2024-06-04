import os
import sys
import json
import numpy as np
import random
import nltk
np.set_printoptions(threshold=np.inf)

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
torch.set_printoptions(profile="full")

from mingpt.utils import set_seed, setup_logging, CfgNode as CN

sort_methods = (
    (False, lambda train_data: sum([(10**(9-i))*train_data[i] for i in range(10)])) 
    ,(False, lambda train_data: 100*train_data[0]+10*train_data[1]+train_data[2])
    ,(False, lambda train_data: 100*train_data[3]+10*train_data[4]+train_data[5])
    ,(False, lambda train_data: 10*train_data[6]+train_data[7])  
    ,(False, lambda train_data: 10*train_data[8]+train_data[9])
    ,(True, lambda train_data: 100*train_data[0]+10*train_data[1]+train_data[2])
    ,(True, lambda train_data: 100*train_data[3]+10*train_data[4]+train_data[5])
    ,(True, lambda train_data: 10*train_data[6]+train_data[7])  
    ,(True, lambda train_data: 10*train_data[8]+train_data[9])
)

### this is done by myself
def shuffle(l):
    n = len(l)
    if n<=1:
        return l
    mid = n // 2
    ll = l[0:mid]
    lr = l[mid:n]
    shuffle(ll)
    shuffle(lr)
    l = ll + lr
    return l


### end 

class ChickenRabbitDataset(Dataset):

    @staticmethod
    def get_default_config():
        C = CN()
        # the digits for the first / second number 5 => 0 0 5, 24 => 0 2 4
        C.ndigit = 3
        return C
    
    def chicken_rabbit(self):
        """
        we only want valid number of chicks or rabbits, so we will first get 
        all the possible combination of c & r and calculate the corresponding 
        heads and legs.
        """
        data = []
        for i in range(100):
            for j in range(100):
                d = f"{i+j:03}{2*i+4*j:03}{i:02}{j:02}"
                data.append([int(dd) for dd in d])
        return data

    def __init__(self, config, split, seed, idx):
        self.config = config
        # split up all addition problems into either training data or test data
        self.split = split # train / test
        data = self.chicken_rabbit()
        print(f'the length of the whole data = {len(data)}')

        random.Random(seed).shuffle(data)
        perm = torch.tensor(data, dtype=torch.long)

        num_test = min(int(len(perm)*0.2), 500) # 20% of the whole dataset, or only up to 500

        train_data = perm[num_test:].tolist()
        train_data.sort(reverse = sort_methods[idx][0], key=sort_methods[idx][1])
        
        ''' updown
        train_zigzag = []
        for i in range(0, len(train_data), 2):
            train_zigzag.append(train_data[i])
        for i in reversed(range(1, len(train_data), 2)):
            train_zigzag.append(train_data[i])
        train_data = train_zigzag
        '''
        shuffle(train_data)
        
        # train_zigzag = [None] * len(train_data)
        # for i in range(len(train_data)):
        #     if i%2==0:
        #         train_zigzag[i]=train_data[i//2]
        #     else:
        #         train_zigzag[i]=train_data[len(train_data) - i//2 -1]
        # train_data = train_zigzag
        
        train_data = torch.tensor(train_data, dtype=torch.long)

        self.ixes = perm[:num_test] if split == 'test' else train_data #perm[num_test:]

    def get_vocab_size(self):
        return 10 # digits 0..9

    def get_block_size(self):
        # in this case: 3 (x+y) + 3 (2x+4y) + 2 (x) + 2 (y) - 1 = 10 - 1
        return 2 * self.config.ndigit + 2 * (self.config.ndigit - 1) - 1

    def __len__(self):
        return self.ixes.size(0)

    def __getitem__(self, idx):
        '''
        logic as follows:
        example data    : [0, 0, 2, 0, 0, 5, 0, 1, 0, 4]
        target          : [0, 2, 0, 0, 5, 0, 1, 0, 4, x]
        expect          : [N, N, N, N, N, 0, 1, 0, 4, x]
        and we will omit the last digit since it's useless
        '''
        # x will be input to GPT and y will be the associated expected outputs
        x = self.ixes[idx][:-1]
        y = self.ixes[idx][1:].clone() # predict the next token in the sequence
        y[:self.config.ndigit * 2 - 1] = -1 # we will only train in the output locations. -1 will mask loss to zero

        return x, y

def eval_split(device, model, dataset):
    ndigit = dataset.config.ndigit
    total_correct = 0
    loader = DataLoader(dataset, shuffle=False, batch_size=100, num_workers=0, drop_last=False)
            
    for b, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        # isolate the first two digits of the input sequence alone
        d1d2 = x[:, :ndigit * 2]
        d3_gt = y[:, ndigit * 2 - 1:]
        # print(f'the shape of d3_gt = {d3_gt.shape}')
        d1d2d3 = model.generate(d1d2, 2 * (ndigit - 1), do_sample=False) # using greedy argmax, not sampling
        d3_pred = d1d2d3[:, ndigit * 2:]
        # print(f'the shape of d3_pred = {d3_pred.shape}')

        # evaluate the correctness of the results in this batch
        correct = torch.sum(torch.eq(torch.sum(d3_pred == d3_gt, dim=1), ndigit * 2 - 2)).item()
        total_correct += correct 

    return total_correct / len(dataset)