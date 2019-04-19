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
from extract_inter_base_model_batch import CONFIG_NAME, DEFAULT_PREDICTORS, compute_metrics, load_model, load_dataset_reader, move_input_to_device
import IPython

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_file", help="Path to model weights file")
    parser.add_argument("--serialization_dir", help="Path to serialization directory")
    parser.add_argument("--val_filepath", help="Path to dataset to evaluate and save intermediate outputs of")
    parser.add_argument("--output_dir", help="Path to output directory")
    parser.add_argument("--cuda", help="CUDA device #",type=int, default=-1)
    args = parser.parse_args()

    # Load model and dataset reader
    model = load_model(args.weights_file, args.serialization_dir, args.cuda)
    dataset_reader = load_dataset_reader(args.serialization_dir)

    # Set cuda device, if available or set
    device = torch.device(args.cuda)

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

        pbar = ProgressBar()
        
        for instance in pbar(val_dataset):
            # Create batch and index instance
            instance_list = [instance]
            dataset = Batch(instance_list)
            dataset.index_instances(model.vocab)
            
            # Change dataset to tensors and predict with model
            model_input = dataset.as_tensor_dict()
            model_input['question'] = {k: v.to(device) for k,v in model_input['question'].items()}
            model_input['passage'] = {k: v.to(device) for k,v in model_input['passage'].items()}
            model_input['span_start'] = model_input['span_start'].to(device)
            model_input['span_end'] = model_input['span_end'].to(device)

            model_outputs = model(**model_input)
            IPython.embed()
            metrics = compute_metrics(model_outputs, **model_input)

            span_start_acc = metrics['span_start_acc']
            span_end_acc = metrics['span_end_acc']

            # Save outputs
            outputs.append(model_outputs)

            # Save in 4 categories/folders
            if span_start_acc and span_end_acc:
                correct_outputs.append(model_outputs)
            
            elif span_start_acc and not span_end_acc:
                correct_start_outputs.append(model_outputs)

            elif not span_start_acc and span_end_acc:
                correct_end_outputs.append(model_outputs)
            
            else:
                incorrect_outputs.append(model_outputs)

        print('Correct: {}, Start Correct: {}, End Correct: {}, Incorrect: {}\n'.format(
              len(correct_outputs), len(correct_start_outputs), len(correct_end_outputs), 
              len(incorrect_outputs)))
        
        # Saving all the intermediate/final inputs/outputs
        torch.save(outputs, os.path.join(dir_name, 'outputs.torch'))
        torch.save(correct_outputs, os.path.join(dir_name, 'correct_outputs.torch'))
        torch.save(correct_start_outputs, os.path.join(dir_name, 'correct_start_outputs.torch'))
        torch.save(correct_end_outputs, os.path.join(dir_name, 'correct_end_outputs.torch'))
        torch.save(incorrect_outputs, os.path.join(dir_name, 'incorrect_outputs.torch'))

    
