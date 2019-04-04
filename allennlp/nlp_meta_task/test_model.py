import torch
import argparse
import os
import json
import logging
from allennlp.models import BidirectionalAttentionFlow
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.predictors import Predictor
from allennlp.data import DatasetReader
from allennlp.data.dataset import Batch
from allennlp.training.metrics import CategoricalAccuracy
from progressbar import ProgressBar
from test_model_batch import CONFIG_NAME, DEFAULT_PREDICTORS, compute_metrics, load_model, load_dataset_reader, move_input_to_device
import IPython

logger = logging.getLogger(__name__)

### HOOKS
ll_start_output = []
ll_end_output = []
model_layer_input = []
model_layer_output = []

def ll_start_hook(self, input, output):
    ll_start_output.append(output)

def ll_end_hook(self, input, output):
    ll_end_output.append(output)

def model_layer_hook(self, input, output):
    model_layer_input.append(input)
    model_layer_output.append(output)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file")
    parser.add_argument("--serialization_dir")
    parser.add_argument("--val_filepath")
    parser.add_argument("--output_dir")
    parser.add_argument("--cuda", type=int, default=-1)
    args = parser.parse_args()

    # Load model and dataset reader
    model = load_model(args.serialization_dir, args.cuda)
    dataset_reader = load_dataset_reader(args.serialization_dir)

    # Attaching hook to:
    # Last two linear layers for span_start and span_end
    # Model layer of LSTM
    layer_list = list(model.children())
    layer_list[-2].register_forward_hook(ll_end_hook)
    layer_list[-3].register_forward_hook(ll_start_hook)
    layer_list[-4].register_forward_hook(model_layer_hook)

    # Set model to evaluation mode
    model.eval()

    val_dataset = dataset_reader.read(args.val_filepath)

    # Creating directory for outputs 
    dir_name = args.output_dir
    try:
        os.mkdir(dir_name)
        logger.info("Created directory for outputs")
    except:
        logger.error("ERROR: Could not create outputs directory")
        raise Exception("Output directory already exists!")

    count = 0
    with torch.no_grad():
        outputs = []
        correct_outputs = []
        correct_start_outputs = []
        correct_end_outputs = []
        incorrect_outputs = []

        # Only correct and incorrect for last layers b/c start and end are separated
        correct_ll_start_outputs = []
        correct_start_ll_start_outputs = []
        correct_end_ll_start_outputs = []
        incorrect_ll_start_outputs = []

        correct_ll_end_outputs = []
        correct_start_ll_end_outputs = []
        correct_end_ll_end_outputs = []
        incorrect_ll_end_outputs = []

        correct_model_layer_inputs = []
        correct_start_model_layer_inputs = []
        correct_end_model_layer_inputs = []
        incorrect_model_layer_inputs = []

        correct_model_layer_outputs = []
        correct_start_model_layer_outputs = []
        correct_end_model_layer_outputs = []
        incorrect_model_layer_outputs = []

        correct_inputs = []
        correct_start_inputs = []
        correct_end_inputs = []
        incorrect_inputs = []

        pbar = ProgressBar()
        
        for instance in pbar(val_dataset):
            # Create batch and index instance
            instance_list = [instance]
            dataset = Batch(instance_list)
            dataset.index_instances(model.vocab)
            
            # Change dataset to tensors and predict with model
            model_input = dataset.as_tensor_dict()
            model_input = move_input_to_device(model_input)
            model_outputs = model(**model_input)
            metrics = compute_metrics(model_outputs, **model_input)

            span_start_acc = metrics['span_start_acc']
            span_end_acc = metrics['span_end_acc']

            # Save outputs
            outputs.append(model_outputs)

            # Save in 4 categories/folders
            if span_start_acc and span_end_acc:
                correct_outputs.append(model_outputs)
                correct_inputs.append(model_input)
                correct_ll_start_outputs.append(ll_start_output[0])
                correct_ll_end_outputs.append(ll_end_output[0])
                correct_model_layer_inputs.append(model_layer_input[0])
                correct_model_layer_outputs.append(model_layer_output[0])
            
            elif span_start_acc and not span_end_acc:
                correct_start_outputs.append(model_outputs)
                correct_start_inputs.append(model_input)
                correct_start_ll_start_outputs.append(ll_start_output[0])
                correct_start_ll_end_outputs.append(ll_end_output[0])
                correct_start_model_layer_inputs.append(model_layer_input[0])
                correct_start_model_layer_outputs.append(model_layer_output[0])

            elif not span_start_acc and span_end_acc:
                correct_end_outputs.append(model_outputs)
                correct_end_inputs.append(model_input)
                correct_end_ll_start_outputs.append(ll_start_output[0])
                correct_end_ll_end_outputs.append(ll_end_output[0])
                correct_end_model_layer_inputs.append(model_layer_input[0])
                correct_end_model_layer_outputs.append(model_layer_output[0])
            
            else:
                incorrect_outputs.append(model_outputs)
                incorrect_inputs.append(model_input)
                incorrect_ll_start_outputs.append(ll_start_output[0])
                incorrect_ll_end_outputs.append(ll_end_output[0])
                incorrect_model_layer_inputs.append(model_layer_input[0])
                incorrect_model_layer_outputs.append(model_layer_output[0])

            ll_start_output.clear()
            ll_end_output.clear()
            model_layer_input.clear()
            model_layer_output.clear()

        print('Correct: {}, Start Correct: {}, End Correct: {}, Incorrect: {}\n'.format(
              len(correct_outputs), len(correct_start_outputs), len(correct_end_outputs), 
              len(incorrect_outputs)))
        
        # Saving all the intermediate/final inputs/outputs
        torch.save(outputs, os.path.join(dir_name, 'outputs.torch'))
        torch.save(correct_outputs, os.path.join(dir_name, 'correct_outputs.torch'))
        torch.save(correct_start_outputs, os.path.join(dir_name, 'correct_start_outputs.torch'))
        torch.save(correct_end_outputs, os.path.join(dir_name, 'correct_end_outputs.torch'))
        torch.save(incorrect_outputs, os.path.join(dir_name, 'incorrect_outputs.torch'))

        torch.save(correct_inputs, os.path.join(dir_name, 'correct_inputs.torch'))
        torch.save(correct_start_inputs, os.path.join(dir_name, 'correct_start_inputs.torch'))
        torch.save(correct_end_inputs, os.path.join(dir_name, 'correct_end_inputs.torch'))
        torch.save(incorrect_inputs, os.path.join(dir_name, 'incorrect_inputs.torch'))

        torch.save(correct_ll_start_outputs, os.path.join(dir_name, 'correct_ll_start_outputs.torch'))
        torch.save(correct_start_ll_start_outputs, os.path.join(dir_name, 'correct_start_ll_start_outputs.torch'))
        torch.save(correct_end_ll_start_outputs, os.path.join(dir_name, 'correct_end_ll_start_outputs.torch'))
        torch.save(incorrect_ll_start_outputs, os.path.join(dir_name, 'incorrect_ll_start_outputs.torch'))

        torch.save(correct_ll_end_outputs, os.path.join(dir_name, 'correct_ll_end_outputs.torch'))
        torch.save(correct_start_ll_end_outputs, os.path.join(dir_name, 'correct_start_ll_end_outputs.torch'))
        torch.save(correct_end_ll_end_outputs, os.path.join(dir_name, 'correct_end_ll_end_outputs.torch'))
        torch.save(incorrect_ll_end_outputs, os.path.join(dir_name, 'incorrect_ll_end_outputs.torch'))
        
        torch.save(correct_model_layer_inputs, os.path.join(dir_name, 'correct_model_layer_inputs.torch'))
        torch.save(correct_start_model_layer_inputs, os.path.join(dir_name, 'correct_start_model_layer_inputs.torch'))
        torch.save(correct_end_model_layer_inputs, os.path.join(dir_name, 'correct_end_model_layer_inputs.torch'))
        torch.save(incorrect_model_layer_inputs, os.path.join(dir_name, 'incorrect_model_layer_inputs.torch'))

        torch.save(correct_model_layer_outputs, os.path.join(dir_name, 'correct_model_layer_outputs.torch'))
        torch.save(correct_start_model_layer_outputs, os.path.join(dir_name, 'correct_start_model_layer_outputs.torch'))
        torch.save(correct_end_model_layer_outputs, os.path.join(dir_name, 'correct_end_model_layer_outputs.torch'))
        torch.save(incorrect_model_layer_outputs, os.path.join(dir_name, 'incorrect_model_layer_outputs.torch'))


    
