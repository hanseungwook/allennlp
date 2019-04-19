# Github Repository
https://github.com/hanseungwook/allennlp

# VM Configuration
https://cloud.google.com/deep-learning-vm/docs/pytorch_start_instance   
Use the below deployment with the following configurations:    
  
8 vCPUs (52GB memory, n1-highmem-8)  
1 or 2 GPUs (NVIDIA Tesla K80)  
Framework: Pytorch 1.0 + fastai 1.0 (CUDA 10.0)  
Check 'Install NVIDIA GPU driver automatically on first startup'  

# 1. Splitting up data for different usages / Extracting data from Squad 2.0
```bash
cd allennlp/allennlp/nlp_meta_task/
python3 split_data.py 
```

```bash
cd allennlp/allennlp/nlp_meta_task/
python3 extract_squad2.0.py --dataset_filepath={relative path to dataset to extract from} --output_file={name of output file to save extracted dataset}
```

Example command
```bash
python3 extract_squad2.0.py --dataset_filepath=../../../squad_datasets/train-v2.0.json --output_file=squad_v2.0_train_pos.json
```

# 2. Running the base model to retrieve intermediate outputs
## Installing AllenNLP Library
```bash
pip install allennlp --user
```

## Installing NVIDIA apex library for faster computation on AllenNLP
https://github.com/NVIDIA/apex

## Obtaining serialized models and squad datasets
Retrieve nlp output datasets from our Google Drive shared folder:

```
CML Research/serialized_models/  
CML Research/squad_datasets
```

## Running the base model
This assumes that you have cloned my **forked** AllenNLP git repository.  
  
```bash
cd allennlp/allennlp/nlp_meta_task/
python3 extract_inter_base_model.py --weights_dir={relative path to the weights to load} --serialization_dir={relative path to serialized_dir} --val_filepath={relative path to squad dataset to evaluate} --cuda={cuda device num or cpu}
```

If you need to separate the intermediate outputs into batches b/c of memory error, the following code:
```bash
python3 extract_inter_base_model_batch.py --weights_file={relative path to the weights to load} --serialization_dir={relative path to serialized_dir} --val_filepath={relative path to squad dataset to evaluate} --output_dir={relative path to output folder to create} --cuda={cuda device num or cpu}
```

Example command
```bash
python3 extract_inter_base_model.py --weights_file=../../../serialized_models/test4/best.th --serialization_dir=../../../serialized_models/test4/ --val_filepath../../../squad_datasets/dataset_val_q.json --output_dir=../../../outputs/train_outputs/ --cuda=0
```  

**Note**: Depending on the existence of GPU, need to change the 'cuda' parameter in test_model.py  

# 2. Running the meta model

## Obtaining nlp output dataset
Retrieve nlp output datasets from our Google Drive shared folder:

```
CML Research/nlp_outputs/meta_outputs  
CML Research/nlp_outputs/val_outputs
```

## Running the meta network
```bash
cd allennlp/allennlp/nlp_meta_task/
python3 nlp_meta_pipeline.py --training_dir={relative path to training data folder for meta network} --validation_dir={relative path to validation data for meta network} --results_dir={relative path of folder to which the results will be saved}
```

**Note**: Depending on the existence of GPU, need to change the CUDA_DEVICE parameter in nlp_meta_pipeline.py  
**Note**: For the results_dir, it should be to a path/folder that does not exist since it will be newly created by the program

Example command
```bash
python3 nlp_meta_pipeline.py --training_dir=../../../outputs/meta_outputs/ --validation_dir=../../../outputs/val_outputs/ --results_dir='./results'
```

