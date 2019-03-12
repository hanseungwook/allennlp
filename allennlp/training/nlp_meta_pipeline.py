from __future__ import print_function
import argparse
import numpy as np
import sys
import datetime
import os
from shutil import copyfile
from tensorboard_logger import configure, log_value
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from full_mnist_pipeline import train_meta
from meta_model import FCMetaNet
import IPython


### GLOBAL PARAMETERS
CUDA_DEVICE = 'cuda:-1'
LAYER_NAMES = ['model_layer_inputs.torch', 'model_layer_outputs.torch', 'll_start_outputs.torch', 'll_end_outputs.torch']
CORRECT = 'correct_'
INCORRECT = 'incorrect_'
CORRECT_START = 'correct_start_'
CORRECT_END = 'correct_end_'

class IntermediateLayersInMemoryDataset(Dataset):
    def __init__(self, correct_files=[], correct_start_files=[], correct_end_files=[], incorrect_files=[], percentage = 1.0, one_class = False, transform=None):
        self.correct_running_count = 0
        self.correct_start_running_count = 0
        self.correct_end_running_count = 0
        self.incorrect_running_count = 0

        self.correct_len = 0
        self.correct_start_len = 0
        self.correct_end_len = 0
        self.incorrect_len = 0

        self.X_data = []
        correct_data = []
        correct_start_data = []
        correct_end_data = []
        incorrect_data = []
        self.dim_size = 0
        self.data_class = None

        # Defining and setting up for respective class of dataset
        if not one_class:
            for i in range(len(correct_files)):
                self.X_data.append([])
            assert len(correct_files) == len(correct_start_files) == len(correct_end_files) == len(incorrect_files)
            self.data_class = 'all'
            
        elif one_class == 'both':
            for i in range(len(correct_files)):
                self.X_data.append([])
            assert len(correct_files) == len(incorrect_files)
            self.data_class = 'both'

        elif one_class == 'correct':
            for i in range(len(correct_files)):
                self.X_data.append([])
            self.data_class = 'correct'

        elif one_class == 'correct_start':
            for i in range(len(correct_start_files)):
                self.X_data.append([])
            self.data_class = 'correct_start'

        elif one_class == 'correct_end':
            for i in range(len(correct_end_files)):
                self.X_data.append([])
            self.data_class = 'correct_end'

        elif one_class == 'incorrect':
            for i in range(len(incorrect_files)):
                self.X_data.append([])
            self.data_class = 'incorrect'

        else:
            raise Exception('one_class must be False, correct, correct_start, correct_end, or incorrect')

        # Loading and adding correct intermediates/outputs to data
        if len(correct_files) > 0:
            loaded = torch.load(correct_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_correct = np.random.choice(all_indices, size=int(percentage * len(loaded)), replace=False)
            self.correct_len = len(loaded)

        for layer_index in range(len(correct_files)):
            loaded = torch.load(correct_files[layer_index])
            for item_idx in selected_indices_correct:
                self.X_data[layer_index].append(loaded[item_idx])

        # Loading and adding correct start intermediates/outputs to data
        if len(correct_start_files) > 0:
            loaded = torch.load(correct_start_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_correct_start = np.random.choice(all_indices, size=int(percentage * len(loaded)), replace = False)
            self.correct_start_len = len(loaded)

        for layer_index in range(len(correct_start_files)):
            loaded = torch.load(correct_start_files[layer_index])
            for item_idx in selected_indices_correct_start:
                self.X_data[layer_index].append(loaded[item_idx])

        # Loading and adding correct end intermediates/outputs to data
        if len(correct_end_files) > 0:
            loaded = torch.load(correct_end_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_correct_end = np.random.choice(all_indices, size=int(percentage * len(loaded)), replace = False)
            self.correct_end_len = len(loaded)

        for layer_index in range(len(correct_end_files)):
            loaded = torch.load(correct_end_files[layer_index])
            for item_idx in correct_end_files:
                self.X_data[layer_index].append(loaded[item_idx])

        # Loading and adding incorrect intermediates/outputs to data
        if len(incorrect_files) > 0:
            loaded = torch.load(incorrect_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_incorrect = np.random.choice(all_indices, size=int(percentage * len(loaded)), replace = False)
            self.incorrect_len = len(loaded)

        for layer_index in range(len(incorrect_files)):
            loaded = torch.load(incorrect_files[layer_index])
            for item_idx in selected_indices_incorrect:
                self.X_data[layer_index].append(loaded[item_idx])

        #IPython.embed()
        self.dim_size = self.X_data[0][0][0].shape[1]
        self.total_len = self.correct_len + self.correct_start_len + \
                         self.correct_end_len + self.incorrect_len

        # Create labels
        ### TODO: Only accounts for both correct and incorrect, without consideration of correct start and correct end
        self.Y_data = np.zeros((self.total_len))
        self.Y_data[0:self.correct_len] = 1

        print('Total: {}, Correct: {}, Start Correct: {}, End Correct: {}, Incorrect: {}, Percentage of Incorrect: {}\n'.format(
              self.total_len, self.correct_len, self.correct_start_len, self.correct_end_len, self.incorrect_len,
              (self.incorrect_len / self.total_len)))

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        Xs_to_return = []

        for layer in range(len(self.X_data)):
            Xs_to_return.append(self.X_data[layer][idx].float().to(CUDA_DEVICE))

        if self.Y_data[idx] == 1:
            self.correct_running_count += 1
        else:
            self.incorrect_running_count += 1

        return (Xs_to_return, torch.tensor(self.Y_data[idx]).long())

    def get_correct_len(self):
        return self.correct_len

    def get_incorrect_len(self):
        return self.incorrect_len

    def get_correct_start_len(self):
        return self.correct_start_len

    def get_correct_end_len(self):
        return self.correct_end_len

    def get_y_data(self):
        return self.Y_data

    def get_num_layers(self): 
        return self.X_data.shape()[0]
    
    def get_size(self):
        return self.dim_size

def make_and_train_meta_model(args, device, train_set_percentage):
    correct_files = []
    incorrect_files = []
    
    # Compiling list of files of intermediate inputs/outputs
    for layer in LAYER_NAMES:
        correct_files.append(args.result_folder + CORRECT + layer)
        incorrect_files.append(args.result_folder + INCORRECT + layer)
    
    # Create training dataset
    train_dataset = IntermediateLayersInMemoryDataset(correct_files=correct_files, incorrect_files=incorrect_files, one_class='both')
    
    IPython.embed()
    # Get counts and make weights for balancing training samples
    correct_count = train_dataset.get_correct_len()
    incorrect_count = train_dataset.get_incorrect_len()
    total_count = correct_count + incorrect_count
    
    y_vals = train_dataset.get_y_data()

    correct_weight = float(total_count)/correct_count
    incorrect_weight = float(total_count)/incorrect_count

    weights = np.zeros((total_count))

    for i in range(len(y_vals)):
        if y_vals[i] == 0:
            weights[i] = incorrect_weight
        else:
            weights[i] = correct_weight
    
    correct_range = list(range(correct_count))
    incorrect_range = list(range(correct_count, total_count))
    total_range = list(range(total_count))

    # Creating data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,  shuffle = False,
        sampler = train_weighted_sampler)
    # error_validation_loader = torch.utils.data.DataLoader(valid_error_dataset, batch_size = 32,  shuffle = False)
    # correct_validation_loader = torch.utils.data.DataLoader(valid_correct_dataset, batch_size = 32,  shuffle = False)
    # error_test_loader = torch.utils.data.DataLoader(test_error_dataset, batch_size = 32,  shuffle = False)
    # correct_test_loader = torch.utils.data.DataLoader(test_correct_dataset, batch_size = 32,  shuffle = False)
    train_weights = torch.DoubleTensor(weights)
    train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, total_count)

    # Setting seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Setting up meta model
    size_of_first_layer = train_dataset.get_size()
    meta_model=FCMetaNet(size_of_first_layer).to(device)

    # If saved state given, load
    if args.load_meta_model_from_saved_state:
        meta_saved_state = torch.load(args.load_meta_model_from_saved_state)
        meta_model.load_state_dict(meta_saved_state)
    

def main():
    parser = argparse.ArgumentParser(description='Meta NLP pipeline')
    parser.add_argument('result_folder', help='folder in which the intermediate inputs/outputs are saved')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',help='input batch size for testing (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=10027, metavar='S',
                        help='random seed (default: 10027)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--meta_batch_size', type=int, default=1, metavar='MBS',
                        help='size of batches to the meta classifier')
    parser.add_argument('--meta_train_num_epochs', type=int, default=50, metavar='metatrainepochs',
                        help='size of batches to the meta classifier')
    parser.add_argument('--load_meta_model_from_saved_state', default="")

    args = parser.parse_args()
    make_and_train_meta_model(args, -1, 1.0)

if __name__ == "__main__":
    main()