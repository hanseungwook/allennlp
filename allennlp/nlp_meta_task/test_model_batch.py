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

import IPython

logger = logging.getLogger(__name__)

### GLOBAL CONSTANT VARIABLES
CONFIG_NAME = "config.json"
DEFAULT_PREDICTORS = {
        'atis_parser' : 'atis_parser',
        'biaffine_parser': 'biaffine-dependency-parser',
        'bidaf': 'machine-comprehension',
        'bidaf-ensemble': 'machine-comprehension',
        'bimpm': 'textual-entailment',
        'constituency_parser': 'constituency-parser',
        'coref': 'coreference-resolution',
        'crf_tagger': 'sentence-tagger',
        'decomposable_attention': 'textual-entailment',
        'dialog_qa': 'dialog_qa',
        'event2mind': 'event2mind',
        'simple_tagger': 'sentence-tagger',
        'srl': 'semantic-role-labeling',
        'quarel_parser': 'quarel-parser',
        'wikitables_mml_parser': 'wikitables-parser'
}
INTER_NAMES = ['outputs', 'correct_outputs', 'correct_start_outputs', 'correct_end_outputs', 'incorrect_outputs',
                'correct_inputs', 'correct_start_inputs', 'correct_end_inputs', 'incorrect_inputs',
                'correct_ll_start_outputs', 'correct_start_ll_start_outputs', 'correct_end_ll_start_outputs', 'incorrect_ll_start_outputs',
                'correct_ll_end_outputs', 'correct_start_ll_end_outputs', 'correct_end_ll_start_outputs', 'incorrect_ll_end_outputs',
                'correct_model_layer_inputs', 'correct_start_model_layer_inputs', 'correct_end_model_layer_inputs', 'incorrect__model_layer_inputs',
                'correct_model_layer_outputs', 'correct_start_model_layer_outputs', 'correct_end_model_layer_outputs', 'incorrect_model_layer_outputs']       

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

# Function for calculating span_start_accuracy and span_end_accuracy
def compute_metrics(outputs, question, passage, span_start = None, span_end = None, metadata = None):
    metrics = {}    

    span_start_acc = CategoricalAccuracy()
    span_start_acc(outputs['span_start_logits'], span_start.squeeze(-1))
    metrics['span_start_acc'] = span_start_acc.get_metric()

    span_end_acc = CategoricalAccuracy()
    span_end_acc(outputs['span_end_logits'], span_end.squeeze(-1))
    metrics['span_end_acc'] = span_end_acc.get_metric()
    return metrics

def load_model(serialization_dir, cuda):
    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME))
    config.loading_from_archive = True
    model = Model.load(config.duplicate(),
                    weights_file = args.weights_file,
                    serialization_dir = args.serialization_dir,
                    cuda_device = cuda)

    return model

def load_dataset_reader(serialization_dir):
    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME))
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    return dataset_reader

# Create nested directory structure for output
def make_output_dirs(output_dir):
    # Create parent output directory
    try:
        os.mkdir(output_dir)
        logger.info("Created directory for outputs")
    except:
        logger.error("ERROR: Could not create outputs directory")

    # Create nested output directories
    for inter in INTER_NAMES:
        inter_folder = os.path.join(output_dir, inter)
        if not os.path.exists(inter_folder):
            os.mkdir(inter_folder)

def move_input_to_device(model_input):
    model_input['question'] = {k: v.to(device) for k,v in model_input['question'].items()}
    model_input['passage'] = {k: v.to(device) for k,v in model_input['passage'].items()}
    model_input['span_start'] = model_input['span_start'].to(device)
    model_input['span_end'] = model_input['span_end'].to(device)

    return model_input

# logger.info("Read {} test examples".format(len(val_dataset.instances)))

### PREDICTION OF 1 EXAMPLE
# model_type = config.get("model").get("type")
# if not model_type in DEFAULT_PREDICTORS:
#     raise ConfigurationError(f"No default predictor for model type {model_type}.\n"\
#                                 f"Please specify a predictor explicitly.")
# predictor_name = DEFAULT_PREDICTORS[model_type]

# model_predictor = Predictor.by_name(predictor_name)(model, dataset_reader)
# prediction = model_predictor.predict("Which Secretary of State attended Notre Dame?",
#                                      "Notre Dame alumni work in various fields. Alumni working in political fields include state governors, members of the United States Congress, and former United States Secretary of State Condoleezza Rice. A notable alumnus of the College of Science is Medicine Nobel Prize winner Eric F. Wieschaus. A number of university heads are alumni, including Notre Dame's current president, the Rev. John Jenkins. Additionally, many alumni are in the media, including talk show hosts Regis Philbin and Phil Donahue, and television and radio personalities such as Mike Golic and Hannah Storm. With the university having high profile sports teams itself, a number of alumni went on to become involved in athletics outside the university, including professional baseball, basketball, football, and ice hockey players, such as Joe Theismann, Joe Montana, Tim Brown, Ross Browner, Rocket Ismail, Ruth Riley, Jeff Samardzija, Jerome Bettis, Brett Lebda, Olympic gold medalist Mariel Zagunis, professional boxer Mike Lee, former football coaches such as Charlie Weis, Frank Leahy and Knute Rockne, and Basketball Hall of Famers Austin Carr and Adrian Dantley. Other notable alumni include prominent businessman Edward J. DeBartolo, Jr. and astronaut Jim Wetherbee.")

# with open('test_prediction.json', 'w') as predict_file:
#     json.dump(prediction, predict_file)
def move_input_to_device(model_input):
    
    return model_input


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

    # Set cuda device, if available or set
    device = torch.device(args.cuda)

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

    # Create nested directories for outputs
    dir_name = args.output_dir
    make_output_dirs(dir_name)
    
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
        instance_count = 0
        batch_index = 0
        

        for instance in pbar(val_dataset):
            # Increment instance count
            instance_count += 1

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

           # model_input = move_input_to_device(model_input)

            model_outputs = model(**model_input)
            metrics = compute_metrics(model_outputs, **model_input)

            span_start_acc = metrics['span_start_acc']
            span_end_acc = metrics['span_end_acc']

            # Save outputs
            outputs.append(model_outputs)

            model_layer_input[0] = tuple(elem.cpu() for elem in model_layer_input[0])
            ll_start_output[0] = ll_start_output[0].cpu() 
            ll_end_output[0] = ll_end_output[0].cpu()
            model_layer_output[0] = model_layer_output[0].cpu()

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


            if instance_count % 10000 == 0:
                batch_index += 1

                # Saving all the intermediate/final inputs/outputs
                torch.save(outputs, os.path.join(dir_name, INTER_NAMES[0], 'outputs{}.torch'.format(batch_index)))
                torch.save(correct_outputs, os.path.join(dir_name, INTER_NAMES[1], 'correct_outputs{}.torch'.format(batch_index)))
                torch.save(correct_start_outputs, os.path.join(dir_name, INTER_NAMES[2], 'correct_start_outputs{}.torch'.format(batch_index)))
                torch.save(correct_end_outputs, os.path.join(dir_name, INTER_NAMES[3], 'correct_end_outputs{}.torch'.format(batch_index)))
                torch.save(incorrect_outputs, os.path.join(dir_name, INTER_NAMES[4], 'incorrect_outputs{}.torch'.format(batch_index)))

                torch.save(correct_inputs, os.path.join(dir_name, INTER_NAMES[5], 'correct_inputs{}.torch'.format(batch_index)))
                torch.save(correct_start_inputs, os.path.join(dir_name, INTER_NAMES[6], 'correct_start_inputs{}.torch'.format(batch_index)))
                torch.save(correct_end_inputs, os.path.join(dir_name, INTER_NAMES[7], 'correct_end_inputs{}.torch'.format(batch_index)))
                torch.save(incorrect_inputs, os.path.join(dir_name, INTER_NAMES[8], 'incorrect_inputs{}.torch'.format(batch_index)))

                torch.save(correct_ll_start_outputs, os.path.join(dir_name, INTER_NAMES[9], 'correct_ll_start_outputs{}.torch'.format(batch_index)))
                torch.save(correct_start_ll_start_outputs, os.path.join(dir_name, INTER_NAMES[10], 'correct_start_ll_start_outputs{}.torch'.format(batch_index)))
                torch.save(correct_end_ll_start_outputs, os.path.join(dir_name, INTER_NAMES[11], 'correct_end_ll_start_outputs{}.torch'.format(batch_index)))
                torch.save(incorrect_ll_start_outputs, os.path.join(dir_name, INTER_NAMES[12], 'incorrect_ll_start_outputs{}.torch'.format(batch_index)))

                torch.save(correct_ll_end_outputs, os.path.join(dir_name, INTER_NAMES[12], 'correct_ll_end_outputs{}.torch'.format(batch_index)))
                torch.save(correct_start_ll_end_outputs, os.path.join(dir_name, INTER_NAMES[13], 'correct_start_ll_end_outputs{}.torch'.format(batch_index)))
                torch.save(correct_end_ll_end_outputs, os.path.join(dir_name, INTER_NAMES[14], 'correct_end_ll_end_outputs{}.torch'.format(batch_index)))
                torch.save(incorrect_ll_end_outputs, os.path.join(dir_name, INTER_NAMES[15], 'incorrect_ll_end_outputs{}.torch'.format(batch_index)))
                
                torch.save(correct_model_layer_inputs, os.path.join(dir_name, INTER_NAMES[16], 'correct_model_layer_inputs{}.torch'.format(batch_index)))
                torch.save(correct_start_model_layer_inputs, os.path.join(dir_name, INTER_NAMES[17], 'correct_start_model_layer_inputs{}.torch'.format(batch_index)))
                torch.save(correct_end_model_layer_inputs, os.path.join(dir_name, INTER_NAMES[18], 'correct_end_model_layer_inputs{}.torch'.format(batch_index)))
                torch.save(incorrect_model_layer_inputs, os.path.join(dir_name, INTER_NAMES[19], 'incorrect_model_layer_inputs{}.torch'.format(batch_index)))

                torch.save(correct_model_layer_outputs, os.path.join(dir_name, INTER_NAMES[20], 'correct_model_layer_outputs{}.torch'.format(batch_index)))
                torch.save(correct_start_model_layer_outputs, os.path.join(dir_name, INTER_NAMES[21], 'correct_start_model_layer_outputs{}.torch'.format(batch_index)))
                torch.save(correct_end_model_layer_outputs, os.path.join(dir_name, INTER_NAMES[22], 'correct_end_model_layer_outputs{}.torch'.format(batch_index)))
                torch.save(incorrect_model_layer_outputs, os.path.join(dir_name, INTER_NAMES[23], 'incorrect_model_layer_outputs{}.torch'.format(batch_index)))

                # Clearing intermediate outputs
                outputs.clear()
                correct_outputs.clear()
                correct_start_outputs.clear()
                correct_end_outputs.clear()
                incorrect_outputs.clear()

                correct_ll_start_outputs.clear()
                correct_start_ll_start_outputs.clear()
                correct_end_ll_start_outputs.clear()
                incorrect_ll_start_outputs.clear()

                correct_ll_end_outputs.clear()
                correct_start_ll_end_outputs.clear()
                correct_end_ll_end_outputs.clear()
                incorrect_ll_end_outputs.clear()

                correct_model_layer_inputs.clear()
                correct_start_model_layer_inputs.clear()
                correct_end_model_layer_inputs.clear()
                incorrect_model_layer_inputs.clear()

                correct_model_layer_outputs.clear()
                correct_start_model_layer_outputs.clear()
                correct_end_model_layer_outputs.clear()
                incorrect_model_layer_outputs.clear()

                correct_inputs.clear()
                correct_start_inputs.clear()
                correct_end_inputs.clear()
                incorrect_inputs.clear()

        # Saving last batch / leftovers
        batch_index += 1

        # Saving all the intermediate/final inputs/outputs
        torch.save(outputs, os.path.join(dir_name, 'outputs{}.torch'.format(batch_index)))
        torch.save(correct_outputs, os.path.join(dir_name, 'correct_outputs{}.torch'.format(batch_index)))
        torch.save(correct_start_outputs, os.path.join(dir_name, 'correct_start_outputs{}.torch'.format(batch_index)))
        torch.save(correct_end_outputs, os.path.join(dir_name, 'correct_end_outputs{}.torch'.format(batch_index)))
        torch.save(incorrect_outputs, os.path.join(dir_name, 'incorrect_outputs{}.torch'.format(batch_index)))

        torch.save(correct_inputs, os.path.join(dir_name, 'correct_inputs{}.torch'.format(batch_index)))
        torch.save(correct_start_inputs, os.path.join(dir_name, 'correct_start_inputs{}.torch'.format(batch_index)))
        torch.save(correct_end_inputs, os.path.join(dir_name, 'correct_end_inputs{}.torch'.format(batch_index)))
        torch.save(incorrect_inputs, os.path.join(dir_name, 'incorrect_inputs{}.torch'.format(batch_index)))

        torch.save(correct_ll_start_outputs, os.path.join(dir_name, 'correct_ll_start_outputs{}.torch'.format(batch_index)))
        torch.save(correct_start_ll_start_outputs, os.path.join(dir_name, 'correct_start_ll_start_outputs{}.torch'.format(batch_index)))
        torch.save(correct_end_ll_start_outputs, os.path.join(dir_name, 'correct_end_ll_start_outputs{}.torch'.format(batch_index)))
        torch.save(incorrect_ll_start_outputs, os.path.join(dir_name, 'incorrect_ll_start_outputs{}.torch'.format(batch_index)))

        torch.save(correct_ll_end_outputs, os.path.join(dir_name, 'correct_ll_end_outputs{}.torch'.format(batch_index)))
        torch.save(correct_start_ll_end_outputs, os.path.join(dir_name, 'correct_start_ll_end_outputs{}.torch'.format(batch_index)))
        torch.save(correct_end_ll_end_outputs, os.path.join(dir_name, 'correct_end_ll_end_outputs{}.torch'.format(batch_index)))
        torch.save(incorrect_ll_end_outputs, os.path.join(dir_name, 'incorrect_ll_end_outputs{}.torch'.format(batch_index)))
        
        torch.save(correct_model_layer_inputs, os.path.join(dir_name, 'correct_model_layer_inputs{}.torch'.format(batch_index)))
        torch.save(correct_start_model_layer_inputs, os.path.join(dir_name, 'correct_start_model_layer_inputs{}.torch'.format(batch_index)))
        torch.save(correct_end_model_layer_inputs, os.path.join(dir_name, 'correct_end_model_layer_inputs{}.torch'.format(batch_index)))
        torch.save(incorrect_model_layer_inputs, os.path.join(dir_name, 'incorrect_model_layer_inputs{}.torch'.format(batch_index)))

        torch.save(correct_model_layer_outputs, os.path.join(dir_name, 'correct_model_layer_outputs{}.torch'.format(batch_index)))
        torch.save(correct_start_model_layer_outputs, os.path.join(dir_name, 'correct_start_model_layer_outputs{}.torch'.format(batch_index)))
        torch.save(correct_end_model_layer_outputs, os.path.join(dir_name, 'correct_end_model_layer_outputs{}.torch'.format(batch_index)))
        torch.save(incorrect_model_layer_outputs, os.path.join(dir_name, 'incorrect_model_layer_outputs{}.torch'.format(batch_index)))

        # print('Correct: {}, Start Correct: {}, End Correct: {}, Incorrect: {}\n'.format(
        #       len(correct_outputs), len(correct_start_outputs), len(correct_end_outputs), 
        #       len(incorrect_outputs)))
        
        


    
