import torch
import argparse
import os
import json
from allennlp.models import BidirectionalAttentionFlow
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.predictors import Predictor
from allennlp.data import DatasetReader
import IPython

ll_output = []

def ll_hook(self, input, output):
    ll_output.append(output)


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

parser = argparse.ArgumentParser()
parser.add_argument("weights_file")
parser.add_argument("serialization_dir")
parser.add_argument("cuda_device")
args = parser.parse_args()


config = Params.from_file(os.path.join(args.serialization_dir, CONFIG_NAME))
config.loading_from_archive = True

cuda_device = int(args.cuda_device)

model = Model.load(config.duplicate(),
                   weights_file = args.weights_file,
                   serialization_dir = args.serialization_dir,
                   cuda_device = cuda_device)

# Printing model's children
# for module in model.children():
#     print(type(module))
#     print(module)

# Attaching hook to last layer
modules_list = list(model.children())
modules_list[-2].register_forward_hook(ll_hook)

dataset_reader_params = config["dataset_reader"]
dataset_reader = DatasetReader.from_params(dataset_reader_params)

model.eval()

model_type = config.get("model").get("type")
if not model_type in DEFAULT_PREDICTORS:
    raise ConfigurationError(f"No default predictor for model type {model_type}.\n"\
                                f"Please specify a predictor explicitly.")
predictor_name = DEFAULT_PREDICTORS[model_type]

model_predictor = Predictor.by_name(predictor_name)(model, dataset_reader)
prediction = model_predictor.predict("Which Secretary of State attended Notre Dame?",
                                     "Notre Dame alumni work in various fields. Alumni working in political fields include state governors, members of the United States Congress, and former United States Secretary of State Condoleezza Rice. A notable alumnus of the College of Science is Medicine Nobel Prize winner Eric F. Wieschaus. A number of university heads are alumni, including Notre Dame's current president, the Rev. John Jenkins. Additionally, many alumni are in the media, including talk show hosts Regis Philbin and Phil Donahue, and television and radio personalities such as Mike Golic and Hannah Storm. With the university having high profile sports teams itself, a number of alumni went on to become involved in athletics outside the university, including professional baseball, basketball, football, and ice hockey players, such as Joe Theismann, Joe Montana, Tim Brown, Ross Browner, Rocket Ismail, Ruth Riley, Jeff Samardzija, Jerome Bettis, Brett Lebda, Olympic gold medalist Mariel Zagunis, professional boxer Mike Lee, former football coaches such as Charlie Weis, Frank Leahy and Knute Rockne, and Basketball Hall of Famers Austin Carr and Adrian Dantley. Other notable alumni include prominent businessman Edward J. DeBartolo, Jr. and astronaut Jim Wetherbee.")

with open('test_prediction.json', 'w') as predict_file:
    json.dump(prediction, predict_file)

IPython.embed()

