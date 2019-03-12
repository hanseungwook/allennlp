from __future__ import print_function
import argparse
import logging
import sys
import datetime
import os
import numpy as np
from shutil import copyfile
from tensorboard_logger import configure, log_value
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import default_collate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from meta_model import FCMetaNet, train_meta, test_meta_model
import IPython


### GLOBAL PARAMETERS
CUDA_DEVICE = 'cpu'
# LAYER_NAMES = ['model_layer_inputs.torch', 'model_layer_outputs.torch', 'll_start_outputs.torch', 'll_end_outputs.torch']
LAYER_NAMES = ['ll_start_outputs.torch', 'll_end_outputs.torch']
CORRECT = 'correct_'
INCORRECT = 'incorrect_'
CORRECT_START = 'correct_start_'
CORRECT_END = 'correct_end_'
INPUTS = 'inputs.torch'


# Setting up logger
LOGGER = logging.getLogger(__name__)
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setLevel(logging.INFO)
LOGGER.addHandler(out_hdlr)
LOGGER.setLevel(logging.INFO)

class IntermediateLayersInMemoryDataset(Dataset):
    def __init__(self, correct_files=None, correct_start_files=None, correct_end_files=None, 
                 incorrect_files=None, input_files=None, percentage = 1.0, one_class = False, transform=None):
        self.correct_running_count = 0
        self.correct_start_running_count = 0
        self.correct_end_running_count = 0
        self.incorrect_running_count = 0

        self.correct_len = 0
        self.correct_start_len = 0
        self.correct_end_len = 0
        self.incorrect_len = 0

        self.X_data = []
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
        if correct_files:
            loaded = torch.load(correct_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_correct = np.random.choice(all_indices, size=int(percentage * len(loaded)), 
                                                        replace=False)
            self.correct_len = len(loaded)

            for layer_index in range(len(correct_files)):
                loaded = torch.load(correct_files[layer_index])
                for item_idx in selected_indices_correct:
                    processed_data = process_layer_data(loaded[item_idx], layer_index)
                    self.X_data[layer_index].append(processed_data)

        # Loading and adding correct start intermediates/outputs to data
        if correct_start_files:
            loaded = torch.load(correct_start_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_correct_start = np.random.choice(all_indices, size=int(percentage * len(loaded)),
                                                              replace = False)
            self.correct_start_len = len(loaded)

            for layer_index in range(len(correct_start_files)):
                loaded = torch.load(correct_start_files[layer_index])
                for item_idx in selected_indices_correct_start:
                    processed_data = process_layer_data(loaded[item_idx], layer_index)
                    self.X_data[layer_index].append(processed_data)

        # Loading and adding correct end intermediates/outputs to data
        if correct_end_files:
            loaded = torch.load(correct_end_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_correct_end = np.random.choice(all_indices, size=int(percentage * len(loaded)),
                                                            replace = False)
            self.correct_end_len = len(loaded)

            for layer_index in range(len(correct_end_files)):
                loaded = torch.load(correct_end_files[layer_index])
                for item_idx in selected_indices_correct_end:
                    processed_data = process_layer_data(loaded[item_idx], layer_index)
                    self.X_data[layer_index].append(processed_data)

        # Loading and adding incorrect intermediates/outputs to data
        if incorrect_files:
            loaded = torch.load(incorrect_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_incorrect = np.random.choice(all_indices, size=int(percentage * len(loaded)),
                                                          replace = False)
            self.incorrect_len = len(loaded)

            for layer_index in range(len(incorrect_files)):
                loaded = torch.load(incorrect_files[layer_index])
                for item_idx in selected_indices_incorrect:
                    processed_data = process_layer_data(loaded[item_idx], layer_index)
                    self.X_data[layer_index].append(processed_data)

        # Put padding onto the intermediate layer features
        self.X_data = pad_layers(self.X_data, 0)
        self.X_data = pad_layers(self.X_data, 1)

        # If only span_start and span_end output layers given, then concatenate the tensors into 1
        self.X_data = cat_outputs(self.X_data, [0, 1])

        # Defining dim_size for 1D vector
        self.dim_size = self.X_data[0][0].shape[0]
        self.total_len = self.correct_len + self.correct_start_len + \
                         self.correct_end_len + self.incorrect_len

        # Create labels
        ### TODO: Only accounts for both correct and incorrect, without consideration of correct start and correct end
        self.Y_data = np.zeros((self.total_len))
        self.Y_data[0:self.correct_len] = 1

        # Load inputs
        self.inputs = []
        input_files = []

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

# Processes the data (tensor) of the layer to reshape and etc given the layer number
def process_layer_data(data, layer_no):
    processed_data = data
    if len(LAYER_NAMES) == 2:
        if layer_no == 0 or layer_no == 1:
            processed_data = data.view(data.shape[1])

    elif len(LAYER_NAMES) == 4:
        # Model layer input: Only take the first element of the model layer input tuple of tensors
        if layer_no == 0:
            processed_data = data[0]
        # Output layers: Reshape to make it 1D tensor
        elif layer_no == 2 or layer_no == 3:
            processed_data = data.view(data.shape[1])

    return processed_data

def cat_outputs(X_data, cat_idx_list):

    cat1 = cat_idx_list[0]
    cat2 = cat_idx_list[1]

    for i in range(len(X_data[cat1])):
        X_data[cat1][i] = torch.cat((X_data[cat1][i], X_data[cat2][i]), 0)

    del X_data[cat2]

    return X_data

# Pad data at given layer indices with the maximum length of tensor in the dataset
def pad_layers(X_data, layer_idx):
    dim_list = [item.shape[0] for item in X_data[layer_idx]]
    max_dim = max(dim_list)

    for i in range(len(X_data[layer_idx])):
        cur_item = X_data[layer_idx][i]
        padded = torch.zeros(max_dim)
        cur_dim = cur_item.shape[0]
        padded[:cur_dim] = cur_item
        X_data[layer_idx][i] = padded

    return X_data


def make_and_train_meta_model(args, device, train_set_percentage):
    train_correct_files = []
    train_incorrect_files = []
    valid_correct_files = []
    valid_incorrect_files = []
    train_input_files = []
    valid_input_files = []
    
    # Compiling list of files of intermediate inputs/outputs
    for layer in LAYER_NAMES:
        train_correct_files.append(args.training_dir + CORRECT + layer)
        train_incorrect_files.append(args.training_dir + INCORRECT + layer)
        valid_correct_files.append(args.validation_dir + CORRECT + layer)
        valid_incorrect_files.append(args.validation_dir + INCORRECT + layer)

    # Compiling list of files for inputs
    train_input_files.append(args.training_dir + CORRECT + INPUTS)
    train_input_files.append(args.training_dir + CORRECT_START + INPUTS)
    train_input_files.append(args.training_dir + CORRECT_END + INPUTS)
    train_input_files.append(args.training_dir + INCORRECT + INPUTS)

    valid_input_files.append(args.training_dir + CORRECT + INPUTS)
    valid_input_files.append(args.training_dir + CORRECT_START + INPUTS)
    valid_input_files.append(args.training_dir + CORRECT_END + INPUTS)
    valid_input_files.append(args.training_dir + INCORRECT + INPUTS)

    LOGGER.info('Creating training and validation datasets')

    # Create training dataset
    train_dataset = IntermediateLayersInMemoryDataset(correct_files=train_correct_files, incorrect_files=train_incorrect_files,
                                                      input_files=train_input_files, one_class='both')
    valid_correct_dataset = IntermediateLayersInMemoryDataset(correct_files=valid_correct_files, input_files=valid_input_files,
                                                              one_class='correct')
    valid_incorrect_dataset = IntermediateLayersInMemoryDataset(incorrect_files=valid_incorrect_files, 
                                                                input_files=valid_input_files, one_class='incorrect')

    LOGGER.info('Finished creating training and validation datasets')

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

    train_weights = torch.as_tensor(weights, device=device, dtype=torch.double)
    train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, total_count)

    LOGGER.info('Creating training and validation dataset loaders')
    # Creating data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,  shuffle = False,
        sampler = train_weighted_sampler)
    correct_valid_loader = torch.utils.data.DataLoader(valid_correct_dataset, batch_size = 32,  shuffle = False)
    incorrect_valid_loader = torch.utils.data.DataLoader(valid_incorrect_dataset, batch_size = 32,  shuffle = False)

    # Setting seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    LOGGER.info('Setting up meta network')

    # Setting up meta model
    size_of_first_layer = train_dataset.get_size()
    meta_model=FCMetaNet(size_of_first_layer).to(device)

    # If saved state given, load into model
    if args.load_meta_model_from_saved_state:
        meta_saved_state = torch.load(args.load_meta_model_from_saved_state)
        meta_model.load_state_dict(meta_saved_state)

    try:
        os.mkdir(args.results_dir)
        logger.info("Created directory for outputs")
        accuracies_file_name = os.path.join(results_dir+sys.argv[0]+'_accuracies_record.txt')
        accuracies_file = open(accuracies_file_name, "w+")
    except:
        logger.error('ERROR: Could not create results directory')


    meta_optimizer = optim.Adam(meta_model.parameters(),lr=.00001)
    scheduler = ReduceLROnPlateau(meta_optimizer, 'max', verbose=True)

    best_error_valid_value = 0
    best_correct_valid_value = 0
    best_total_valid_value = 0
    best_total_geo_valid_value = 0
    best_total_diff_adj_geo_acc = 0
    old_diff_adj_geo_acc_file_name_created = False
    old_correct_acc_file_name_created = False
    best_train_acc = 0
    best_total_diff_adj_geo_acc_correct = 0
    best_total_diff_adj_geo_acc_error = 0

    for epoch in range(1,args.meta_train_num_epochs+1):
        train_acc = train_meta(meta_model, device, train_loader, meta_optimizer, epoch)

        correct_acc, error_acc = test_meta_model(meta_model, device, incorrect_validation_loader, correct_validation_loader, meta_optimizer, epoch)
        total_acc = error_acc + correct_acc
        total_geo_acc = np.sqrt(error_acc * correct_acc)
        total_diff_adj_geo_acc = total_geo_acc - np.abs(error_acc-correct_acc)

        accuracies_file.write(str(epoch) + " " + str(train_acc) + " " + " " + str(correct_acc) + " " + str(error_acc)+ " " + str(total_acc) + " " +  str(total_geo_acc) + " " + str(total_diff_adj_geo_acc)+"\n")

        if train_acc > best_train_acc:
            best_train_acc = train_acc

        if total_acc > best_total_valid_value:
            best_total_valid_value = total_acc
            if epoch > 1:
                os.remove(old_total_acc_file_name)

            old_total_acc_file_name = results_folder+'_best_total_acc_valid_epoch_'+str(epoch)+'.pth'
            torch.save(meta_model.state_dict(), old_total_acc_file_name)

        if total_diff_adj_geo_acc > best_total_diff_adj_geo_acc:
            best_total_diff_adj_geo_acc = total_diff_adj_geo_acc
            best_total_diff_adj_geo_acc_correct = correct_acc
            best_total_diff_adj_geo_acc_error = error_acc

            if epoch > 1 and old_diff_adj_geo_acc_file_name_created == True:
                os.remove(old_diff_adj_geo_acc_file_name)

            old_diff_adj_geo_acc_file_name_created = True
            old_diff_adj_geo_acc_file_name = results_folder + '_best_diff_adj_geo_acc_valid_epoch_' + str(epoch) + '.pth'
            torch.save(meta_model.state_dict(), old_diff_adj_geo_acc_file_name)

        
        print("Geo dif adj valid mean acc: " + str(total_diff_adj_geo_acc))

        scheduler.step(total_diff_adj_geo_acc)


    if old_diff_adj_geo_acc_file_name_created == True:
        meta_saved_state = torch.load(old_diff_adj_geo_acc_file_name)
        meta_model.load_state_dict(meta_saved_state)

    #return meta_model, best_total_diff_adj_geo_acc

    # test_correct_acc, test_error_acc = test_meta_model(meta_model, device,error_test_loader, correct_test_loader, meta_optimizer,epoch)
    accuracies_file.close()

    return best_total_diff_adj_geo_acc_correct, best_total_diff_adj_geo_acc_error
    

def main():
    parser = argparse.ArgumentParser(description='Meta NLP pipeline')
    parser.add_argument('--training_dir', help='Folder in which the meta training intermediate inputs/outputs are saved')
    parser.add_argument('--validation_dir', help='Folder in which the meta validation intermedaite inputs/outputs are saved')
    parser.add_argument('--results_dir', default= './', help='Folder in which results will be saved')
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
    parser.add_argument('--meta_train_num_epochs', type=int, default=1, metavar='metatrainepochs',
                        help='size of batches to the meta classifier')
    parser.add_argument('--load_meta_model_from_saved_state', default="")

    args = parser.parse_args()
    device = torch.device(CUDA_DEVICE)

    make_and_train_meta_model(args, device, 1)

if __name__ == "__main__":
    main()