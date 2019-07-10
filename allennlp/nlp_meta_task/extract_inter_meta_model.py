import torch
import argparse
import os
import logging
from extract_inter_base_model_batch import load_model, load_dataset_reader, move_input_to_device
from meta_model import FCMetaNet1
import IPython
from nlp_meta_pipeline import IntermediateLayersInMemoryDataset

logger = logging.getLogger(__name__)

### GLOBAL VAR
DIM_SIZE = 141200
LAYER = 'model_layer_outputs.torch'
CORRECT = 'correct_'
INCORRECT = 'incorrect_'

### HOOKS
fc1_output = []
fc2_output = []
fc3_output = []
fc4_output = []
fc5_output = []
fc6_output = []
fc7_output = []


def fc1_hook(self, input, output):
    fc1_output.append(output)

def fc2_hook(self, input, output):
    fc2_output.append(output)

def fc3_hook(self, input, output):
    fc3_output.append(output)

def fc4_hook(self, input, output):
    fc4_output.append(output)

def fc5_hook(self, input, output):
    fc5_output.append(output)

def fc6_hook(self, input, output):
    fc6_output.append(output)

def fc7_hook(self, input, output):
    fc7_output.append(output)



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_meta_model", help="Path to saved meta model")
    parser.add_argument("--val_filepath", help="Path to dataset to evaluate and save intermediate outputs of")
    parser.add_argument("--output_dir", help="Path to output directory")
    parser.add_argument("--batch_size", help="Batch size for evaluation", type=int, default=16)
    parser.add_argument("--cuda", help="CUDA device #",type=int, default=-1)
    args = parser.parse_args()

    logger.info('Loading model')
    # Load model and dataset reader
    model = FCMetaNet1(DIM_SIZE)
    meta_saved_state = torch.load(args.saved_meta_model)
    model.load_state_dict(meta_saved_state)

    # Set cuda device, if available or set
    device = torch.device(args.cuda)

    logger.info('Attaching hooks')
    # Attaching hook to output of every FC
    layer_list = list(model.children())
    layer_list[0].register_forward_hook(fc1_hook)
    layer_list[2].register_forward_hook(fc2_hook)
    layer_list[4].register_forward_hook(fc3_hook)
    layer_list[6].register_forward_hook(fc4_hook)
    layer_list[8].register_forward_hook(fc5_hook)
    layer_list[10].register_forward_hook(fc6_hook)
    layer_list[12].register_forward_hook(fc7_hook)

    # Set model to evaluation mode
    model.eval()

    logger.info('Creating dataset and loader')
    # Creating dataset and loader
    correct_files = []
    incorrect_files = []
    correct_files.append(args.val_filepath + CORRECT + LAYER)
    incorrect_files.append(args.val_filepath + INCORRECT + LAYER)

    dataset = IntermediateLayersInMemoryDataset(correct_files=correct_files, incorrect_files=incorrect_files)
    dataset_loader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False)

    # Creating directory for outputs 
    dir_name = args.output_dir
    try:
        os.mkdir(dir_name)
        logger.info("Created directory for outputs")
    except:
        logger.error("ERROR: Could not create outputs directory")
        raise Exception("Output directory already exists!")

    with torch.no_grad():
        fc1_outputs = []
        fc2_outputs = []
        fc3_outputs = []
        fc4_outputs = []
        fc5_outputs = []
        fc6_outputs = []
        fc7_outputs = []
        
        logger.info('Evaluating dataset with model')
        for batch_idx, (data, target) in enumerate(dataset_loader):
            # Evaluate through model
            data = data.to(device)
            target = target.to(device)
            output = model(data) 

            IPython.embed()

            # Transferring tensors back to CPU
            fc1_outputs = [output.cpu() for output in fc1_output]
            fc2_outputs = [output.cpu() for output in fc2_output]
            fc3_outputs = [output.cpu() for output in fc3_output]
            fc4_outputs = [output.cpu() for output in fc4_output]
            fc5_outputs = [output.cpu() for output in fc5_output]
            fc6_outputs = [output.cpu() for output in fc6_output]
            fc7_outputs = [output.cpu() for output in fc7_output]

            # Save intermediate outputs
            fc1_outputs.extend(fc1_output)
            fc2_outputs.extend(fc2_output)
            fc3_outputs.extend(fc3_output)
            fc4_outputs.extend(fc4_output)
            fc5_outputs.extend(fc5_output)
            fc6_outputs.extend(fc6_output)
            fc7_outputs.extend(fc7_output)

            fc1_output.clear()
            fc2_output.clear()
            fc3_output.clear()
            fc4_output.clear()
            fc5_output.clear()
            fc6_output.clear()
            fc7_output.clear()

        
        # Saving all the intermediate/final inputs/outputs
        torch.save(fc1_outputs, os.path.join(dir_name, 'fc1_outputs.torch'))
        torch.save(fc2_outputs, os.path.join(dir_name, 'fc2_outputs.torch'))
        torch.save(fc3_outputs, os.path.join(dir_name, 'fc3_outputs.torch'))
        torch.save(fc4_outputs, os.path.join(dir_name, 'fc4_outputs.torch'))
        torch.save(fc5_outputs, os.path.join(dir_name, 'fc5_outputs.torch'))
        torch.save(fc6_outputs, os.path.join(dir_name, 'fc6_outputs.torch'))
        torch.save(fc7_outputs, os.path.join(dir_name, 'fc7_outputs.torch'))
