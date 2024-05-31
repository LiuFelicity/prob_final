import os
import sys
import json
import numpy as np
import random
import nltk
np.set_printoptions(threshold=np.inf)

import torch
from ChickenRabbit import ChickenRabbitDataset, eval_split
# from GCD import GCDDataset, eval_split
from torch.utils.data.dataloader import DataLoader
torch.set_printoptions(profile="full")

from mingpt.model_multiplier import GPT
from mingpt.trainer_multiplier import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from itertools import permutations

#import pickle
# -----------------------------------------------------------------------------
seed_weight = 0

def get_config():
    C = CN()

    # system
    C.system = CN()
    # TODO: random seed for model can be set here
    global seed_weight
    C.system.init_seed = seed_weight  # will change the weight initialization     # try 0~9
    C.system.work_dir = './test'

    # data
    C.data = ChickenRabbitDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'
    
    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.task = "gcd" # or gcd
    return C

def batch_end_callback(trainer, model, train_dataset, test_dataset):
    if trainer.iter_num % 10 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

    if trainer.iter_num % 50 == 0:
        # evaluate both the train and test acc
        model.eval()
        with torch.no_grad():
            train_mean = eval_split(trainer.device, model, train_dataset)
            test_mean  = eval_split(trainer.device, model, test_dataset)
        print(f'the mean of train and test are {train_mean}, {test_mean}')
        # save the model and terminate the training
        if test_mean >= 0.9:
            print(f"reach threshold 0.9 in iteration: {trainer.iter_num}")
            print(f"saving model with test_mean: {test_mean}")
            ckpt_path = os.path.join(f"test/{trainer.config.task}", "model_last.pt")
            torch.save(model.state_dict(), ckpt_path)
            return trainer.iter_num
        # revert model to training mode
        model.train()
    return -1

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    arr = []
    for _idx in range(8):
        # Do it 110 times to avoid repeated seed
        for  seed_weight in range(10):
            config = get_config()
            setup_logging(config)

            # TODO: try different seed for model
            
            set_seed(config.system.init_seed)
            # set_seef(_seed)

            # TODO: try different seed to adjust the data order of train/test-set
            _seed = 0 #random.randint(1, 1000000)
            train_dataset = ChickenRabbitDataset(config.data, split='train', seed=_seed, idx=_idx)
            # with open(f"q2_GCD_sort_by_index_{_idx}_increasing.pickle",'wb') as file:
            #    pickle.dump(train_dataset.ixes, file)
            
            test_dataset  = ChickenRabbitDataset(config.data, split='test', seed=_seed, idx=_idx)
            
            # set the correct vocab size: 10, block size: chickenrabbit -> 10, gcd -> 6
            config.model.vocab_size = 10 # train_dataset.get_vocab_size()
            config.model.block_size = 10 # train_dataset.get_block_size()
            model = GPT(config.model)
            trainer = Trainer(config.trainer, model, train_dataset, test_dataset)
            trainer.set_callback('on_batch_end', batch_end_callback)
            stop_iteration = trainer.run()
            if stop_iteration != -1:
                print(f'The final iteration of this round is {stop_iteration}!')
            else:
                print('It cannot reach 0.9 acc within max_iteration steps...')
            
            # store I need {seed, iteration number}
            f = open(f"q2_CR_sort_by_index_{_idx}.txt", "a")
            f.write(f"{seed_weight}, {stop_iteration}\n")
            f.close()

            arr.append( (seed_weight, stop_iteration) )
        print(arr)
        