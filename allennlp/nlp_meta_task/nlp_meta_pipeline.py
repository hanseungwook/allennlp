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
from meta_model import FCMetaNet, FCMetaNet1, FCMetaNet2, FCMetaNet3, train_meta, test_meta_model
import IPython


### GLOBAL PARAMETERS
# LAYER_NAMES = ['model_layer_inputs.torch', 'model_layer_outputs.torch', 'll_start_outputs.torch', 'll_end_outputs.torch']
LAYER_NAMES = ['ll_start_outputs.torch', 'll_end_outputs.torch']
#LAYER_NAMES = ['ll_end_outputs.torch']
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
                 incorrect_files=None, input_files=None, cor_percentage = 1.0, incor_percentage = 1.0,
                 one_class = False, max_dim=0, transform=None):
        self.correct_running_count = 0
        self.correct_start_running_count = 0
        self.correct_end_running_count = 0
        self.incorrect_running_count = 0

        self.correct_len = 0
        self.correct_start_len = 0
        self.correct_end_len = 0
        self.incorrect_len = 0

        self.X_data = []
        #self.dim_size = 0
        self.data_class = None
        self.max_dim = 0

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
            selected_indices_correct = np.random.choice(all_indices, size=int(cor_percentage * len(loaded)), 
                                                        replace=False)
            self.correct_len = int(cor_percentage * len(loaded))

            for layer_index in range(len(correct_files)):
                loaded = torch.load(correct_files[layer_index])
                for item_idx in selected_indices_correct:
                    processed_data = process_layer_data(loaded[item_idx], layer_index)
                    self.X_data[layer_index].append(processed_data)
            
            del loaded

        # Loading and adding correct start intermediates/outputs to data
        if correct_start_files:
            loaded = torch.load(correct_start_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_correct_start = np.random.choice(all_indices, size=int(incor_percentage * len(loaded)),
                                                              replace = False)
            self.correct_start_len = int(incor_percentage * len(loaded))

            for layer_index in range(len(correct_start_files)):
                loaded = torch.load(correct_start_files[layer_index])
                for item_idx in selected_indices_correct_start:
                    processed_data = process_layer_data(loaded[item_idx], layer_index)
                    self.X_data[layer_index].append(processed_data)
            
            del loaded

        # Loading and adding correct end intermediates/outputs to data
        if correct_end_files:
            loaded = torch.load(correct_end_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_correct_end = np.random.choice(all_indices, size=int(incor_percentage * len(loaded)),
                                                            replace = False)
            self.correct_end_len = int(incor_percentage * len(loaded))

            for layer_index in range(len(correct_end_files)):
                loaded = torch.load(correct_end_files[layer_index])
                for item_idx in selected_indices_correct_end:
                    processed_data = process_layer_data(loaded[item_idx], layer_index)
                    self.X_data[layer_index].append(processed_data)

            del loaded

        # Loading and adding incorrect intermediates/outputs to data
        if incorrect_files:
            loaded = torch.load(incorrect_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_incorrect = np.random.choice(all_indices, size=int(incor_percentage * len(loaded)),
                                                          replace = False)
            self.incorrect_len = int(incor_percentage * len(loaded))

            for layer_index in range(len(incorrect_files)):
                loaded = torch.load(incorrect_files[layer_index])
                for item_idx in selected_indices_incorrect:
                    processed_data = process_layer_data(loaded[item_idx], layer_index)
                    self.X_data[layer_index].append(processed_data)
            
            del loaded

        # Defining dim_size for 1D vector
        #self.dim_size = self.X_data[0][0].shape[0]
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
            Xs_to_return.append(self.X_data[layer][idx].float())

        cur_item = Xs_to_return[0]
        padded = torch.zeros(self.max_dim)
        cur_dim = cur_item.shape[0]
        padded[:cur_dim] = cur_item
        Xs_to_return = (padded)

        if self.Y_data[idx] == 1:
            self.correct_running_count += 1
        else:
            self.incorrect_running_count += 1

        return (Xs_to_return, torch.tensor(self.Y_data[idx]).long())

    def calc_max_dim(self, layer_idx_list):
        for layer_idx in layer_idx_list:
            dim_list = [item.shape[0] for item in self.X_data[layer_idx]]
            self.max_dim = max(max(dim_list), self.max_dim)

        return self.max_dim
        
    # Set maximum dimension for padding
    def set_max_dim(self, max_dim):
        self.max_dim = max_dim

    # Filter out features over the max_dim
    def filter_over_dim(self, layer_idx_list):
        for layer_idx in layer_idx_list:
            over_max_dim_idx = []

            for i in range(len(self.X_data[layer_idx])):
                cur_item = self.X_data[layer_idx][i]
                cur_dim = cur_item.shape[0]

                if cur_dim > self.max_dim:
                    over_max_dim_idx.append(i)
            
            for del_idx in sorted(over_max_dim_idx, reverse=True):
                del self.X_data[layer_idx][del_idx]

    # Concatenate layers
    def concat_layers(self, cat_idx_list=None, max_dim_list=None):
        for layer_idx in cat_idx_list:
            for i in range(len(self.X_data[layer_idx])):
                cur_item = self.X_data[layer_idx][i]
                padded = torch.zeros(max_dim_list[layer_idx])
                cur_dim = cur_item.shape[0]
                padded[:cur_dim] = cur_item
                self.X_data[layer_idx][i] = padded

        # If concatenating layer indices given, then concat
        if cat_idx_list:
            cat1 = cat_idx_list[0]
            cat2 = cat_idx_list[1]

            for i in range(len(self.X_data[cat1])):
                self.X_data[cat1][i] = torch.cat((self.X_data[cat1][i], self.X_data[cat2][i]), 0)

            del self.X_data[cat2]

        return self.X_data[cat_idx_list[0]][0].shape        
        # Set dim_size since new tensors have been created
        # self.dim_size = self.X_data[0][0].shape[0]

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
        return self.max_dim

# Processes the data (tensor) of the layer to reshape and etc given the layer number
def process_layer_data(data, layer_no):
    processed_data = data
    # Model layer input
    if len(LAYER_NAMES) == 1 and LAYER_NAMES[0] == 'model_layer_inputs.torch':
        if layer_no == 0:
            processed_data = data[0].reshape(data[0].shape[0] * data[0].shape[1] * data[0].shape[2])
    
    # Model layer output
    elif len(LAYER_NAMES) == 1 and LAYER_NAMES[0] == 'model_layer_outputs.torch':
        if layer_no == 0:
            processed_data = data.reshape(data.shape[0] * data.shape[1] * data.shape[2])

    # Last layer start or end layers
    elif len(LAYER_NAMES) == 1 and (LAYER_NAMES[0] == 'll_start_outputs.torch' or LAYER_NAMES[0] == 'll_end_outputs.torch'):
        if layer_no == 0 or layer_no == 1:
            processed_data = data.view(data.shape[1])
    
    elif len(LAYER_NAMES) == 2 and LAYER_NAMES[0] == 'll_start_outputs.torch' and LAYER_NAMES[1] == 'll_end_outputs.torch':
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

    # Create training and validation dataset (training, only if we are not loading from saved state)
    if not args.load_meta_model_from_saved_state:
        train_dataset = IntermediateLayersInMemoryDataset(correct_files=train_correct_files, incorrect_files=train_incorrect_files,
                                                      input_files=train_input_files, cor_percentage=args.cor_percentage, 
                                                      incor_percentage=args.incor_percentage, one_class='both')
    valid_correct_dataset = IntermediateLayersInMemoryDataset(correct_files=valid_correct_files, input_files=valid_input_files,
                                                              one_class='correct')
    valid_incorrect_dataset = IntermediateLayersInMemoryDataset(incorrect_files=valid_incorrect_files, 
                                                                input_files=valid_input_files, one_class='incorrect')

    LOGGER.info('Finished creating training and validation datasets')

    # Setting layer idx depending on which intermediate layer
    if LAYER_NAMES[0] == 'model_layer_outputs.torch' or LAYER_NAMES[0] == 'model_layer_inputs.torch':
        layer_idx_list = [0] 
    
    # Creating dataset for last layer start or end outputs 
    elif LAYER_NAMES[0] == 'll_start_outputs.torch' or LAYER_NAMES[0] == 'll_end_outputs.torch':
        layer_idx_list = [0]

    # Creating dataset with concat of last layer start and end
    elif len(LAYER_NAMES) == 2 and LAYER_NAMES[0] == 'll_start_outputs.torch' and LAYER_NAMES[1] == 'll_end_outputs.torch':
        layer_idx_list = [0, 1]

    if args.max_dim > 0:
        max_dim = args.max_dim
    else:
        if len(layer_idx_list) == 1:
            max_dim = max(train_dataset.calc_max_dim(layer_idx_list), valid_correct_dataset.calc_max_dim(layer_idx_list),
                          valid_incorrect_dataset.calc_max_dim(layer_idx_list))
        elif len(layer_idx_list) > 1:
            max_dim_list = []
            for layer_idx in layer_idx_list:
                max_dim = max(train_dataset.calc_max_dim([layer_idx]), valid_correct_dataset.calc_max_dim([layer_idx]),
                              valid_incorrect_dataset.calc_max_dim([layer_idx]))
                max_dim_list.append(max_dim)
            
            
            after_concat1 = train_dataset.concat_layers(cat_idx_list=layer_idx_list, max_dim_list=max_dim_list)
            after_concat2 = valid_correct_dataset.concat_layers(cat_idx_list=layer_idx_list, max_dim_list=max_dim_list)
            after_concat3 = valid_incorrect_dataset.concat_layers(cat_idx_list=layer_idx_list, max_dim_list=max_dim_list)
            
            max_dim = sum(max_dim_list)
            IPython.embed()
            
                
    
    LOGGER.info('Set max_dim to {}'.format(max_dim))


    valid_correct_dataset.set_max_dim(max_dim)
    valid_incorrect_dataset.set_max_dim(max_dim)

    if args.load_meta_model_from_saved_state:
        LOGGER.info('Filtering features that are greater than (only for when loading meta model)')
        #train_dataset.filter_over_dim(layer_idx_list=layer_idx_list)
        valid_correct_dataset.filter_over_dim(layer_idx_list=layer_idx_list)
        valid_incorrect_dataset.filter_over_dim(layer_idx_list=layer_idx_list) 

    if not args.load_meta_model_from_saved_state:
        # Setting maximum dimension and filtering
        train_dataset.set_max_dim(max_dim)
        
        LOGGER.info('Calculating weights')
        # Get counts and make weights for balancing training samples
        correct_count = train_dataset.get_correct_len()
        incorrect_count = train_dataset.get_incorrect_len()
        total_count = correct_count + incorrect_count
        
        y_vals = train_dataset.get_y_data()

        correct_weight = float(total_count)/correct_count
        incorrect_weight = float(total_count)/incorrect_count

        train_weights = torch.zeros(total_count, dtype=torch.double)

        for i in range(len(y_vals)):
            if y_vals[i] == 0:
                train_weights[i] = incorrect_weight
            else:
                train_weights[i] = correct_weight
        
        correct_range = list(range(correct_count))
        incorrect_range = list(range(correct_count, total_count))
        total_range = list(range(total_count))

        train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, total_count)

        LOGGER.info('Creating training and validation dataset loaders')
        # Creating data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.meta_batch_size,  shuffle = False,
            sampler = train_weighted_sampler)

    correct_valid_loader = torch.utils.data.DataLoader(valid_correct_dataset, args.meta_batch_size,  shuffle = False)
    incorrect_valid_loader = torch.utils.data.DataLoader(valid_incorrect_dataset, args.meta_batch_size,  shuffle = False)

    # Setting seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    LOGGER.info('Setting up meta network')

    # Setting up meta model
    if not args.load_meta_model_from_saved_state:
        size_of_first_layer = train_dataset.get_size()
    else:
        size_of_first_layer = args.max_dim
    
    LOGGER.info('meta model size of first layer: {}'.format(size_of_first_layer))
    if args.model_class == 0:
        meta_model = FCMetaNet(size_of_first_layer).to(device)
    elif args.model_class == 1:
        meta_model = FCMetaNet1(size_of_first_layer).to(device)
    elif args.model_class == 2:
        meta_model = FCMetaNet2(size_of_first_layer).to(device) 
    elif args.model_class == 3:
        meta_model = FCMetaNet3(size_of_first_layer).to(device)

    # If saved state given, load into model
    if args.load_meta_model_from_saved_state:
        meta_saved_state = torch.load(args.load_meta_model_from_saved_state)
        meta_model.load_state_dict(meta_saved_state)

    try:
        os.mkdir(args.results_dir)
        LOGGER.info("Created directory for results")
    except:
        LOGGER.error('ERROR: Could not create results directory')
        raise Exception('Could not create results directory')

    # Open results file
    accuracies_file_name = os.path.join(args.results_dir, sys.argv[0]+'_accuracies_record.txt')
    accuracies_file = open(accuracies_file_name, "w+")

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

    # Test for loading meta model from saved state
    if args.load_meta_model_from_saved_state:
        LOGGER.info('Evaluating test dataset')
        epoch = 0
        correct_acc, error_acc = test_meta_model(meta_model, device, incorrect_valid_loader, correct_valid_loader, meta_optimizer, epoch, args.results_dir)

        total_acc = error_acc + correct_acc
        total_geo_acc = np.sqrt(error_acc * correct_acc)
        total_diff_adj_geo_acc = total_geo_acc - np.abs(error_acc-correct_acc)

        accuracies_file.write(str(correct_acc) + " " + str(error_acc)+ " " + str(total_acc) + " " +  str(total_geo_acc) + " " + str(total_diff_adj_geo_acc)+"\n")
        accuracies_file.close()

        return correct_acc, error_acc

    # Train + Test
    for epoch in range(1,args.meta_train_num_epochs+1):
        LOGGER.info('Training: Starting epoch {}'.format(epoch))
        train_acc = train_meta(meta_model, device, train_loader, meta_optimizer, epoch)
        LOGGER.info('Training: Finished epoch {}'.format(epoch))
        LOGGER.info('Testing: Starting epoch {}'.format(epoch))
        correct_acc, error_acc = test_meta_model(meta_model, device, incorrect_valid_loader, correct_valid_loader, meta_optimizer, epoch, args.results_dir)
        LOGGER.info('Testing: Finished epoch {}'.format(epoch))
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

            old_total_acc_file_name = os.path.join(args.results_dir, 'best_total_acc_valid_epoch_'+str(epoch)+'.pth')
            torch.save(meta_model.state_dict(), old_total_acc_file_name)

        if total_diff_adj_geo_acc > best_total_diff_adj_geo_acc:
            best_total_diff_adj_geo_acc = total_diff_adj_geo_acc
            best_total_diff_adj_geo_acc_correct = correct_acc
            best_total_diff_adj_geo_acc_error = error_acc

            if epoch > 1 and old_diff_adj_geo_acc_file_name_created == True:
                os.remove(old_diff_adj_geo_acc_file_name)

            old_diff_adj_geo_acc_file_name_created = True
            old_diff_adj_geo_acc_file_name = os.path.join(args.results_dir, 'best_diff_adj_geo_acc_valid_epoch_' + str(epoch) + '.pth')
            torch.save(meta_model.state_dict(), old_diff_adj_geo_acc_file_name)

        
        print("Geo dif adj valid mean acc: " + str(total_diff_adj_geo_acc))

        scheduler.step(total_diff_adj_geo_acc)


    if old_diff_adj_geo_acc_file_name_created == True:
        meta_saved_state = torch.load(old_diff_adj_geo_acc_file_name)
        meta_model.load_state_dict(meta_saved_state)

    #return meta_model, best_total_diff_adj_geo_acc

    # test_correct_acc, test_error_acc = test_meta_model(meta_model, device,error_test_loader, correct_test_loader, meta_optimizer,epoch)
    accuracies_file.close()

    LOGGER.info('Finished epoch {}'.format(epoch))
    return best_total_diff_adj_geo_acc_correct, best_total_diff_adj_geo_acc_error


def main():
    parser = argparse.ArgumentParser(description='Meta NLP pipeline')
    parser.add_argument('--training_dir', help='Folder in which the meta training intermediate inputs/outputs are saved')
    parser.add_argument('--validation_dir', help='Folder in which the meta validation intermedaite inputs/outputs are saved')
    parser.add_argument('--results_dir', default= './', help='Folder in which results will be saved')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',help='input batch size for testing (default: 1000)')
    parser.add_argument('--test_batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 32)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=10027, metavar='S',
                        help='random seed (default: 10027)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--meta_batch_size', type=int, default=16, metavar='MBS',
                        help='size of batches to the meta classifier (default: 16)')
    parser.add_argument('--meta_train_num_epochs', type=int, default=50, metavar='metatrainepochs',
                        help='size of batches to the meta classifier')
    parser.add_argument('--load_meta_model_from_saved_state', default="")
    parser.add_argument('--cuda', type=int, default=-1,
                        help='CUDA device to use')
    parser.add_argument('--cor_percentage', type=float, default=1.0,
                        help='Proportion of correct labels to include in the meta training dataset')
    parser.add_argument('--incor_percentage', type=float, default=1.0,
                        help='Proportion of incorrect labels to include in the meta training dataset')
    parser.add_argument('--max_dim', type=int, default=0,
                        help='Manually setting the maximum dimension of features / first layer size of meta model')
    parser.add_argument('--model_class', type=int, default=1,
                        help='FCMetaNet class number')
    args = parser.parse_args()
    device = torch.device(args.cuda)

    make_and_train_meta_model(args, device, 1)

if __name__ == "__main__":
    main()
