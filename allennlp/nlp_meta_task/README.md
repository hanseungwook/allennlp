# Github Repository
https://github.com/hanseungwook/allennlp

# VM Configuration
https://cloud.google.com/deep-learning-vm/docs/pytorch_start_instance   
Use the below deployment with the following configurations:    
  
8 vCPUs (52GB memory, n1-highmem-8)  
1 or 2 GPUs (NVIDIA Tesla K80)  
Framework: Pytorch 1.0 + fastai 1.0 (CUDA 10.0)  
Check 'Install NVIDIA GPU driver automatically on first startup'  

# 1. Running the base model to retrieve intermediate outputs
## Installing AllenNLP Library
```bash
pip install allennlp --user
```

## Installing NVIDIA apex library for faster computation on AllenNLP
https://github.com/apex/apex  

## Obtaining serialized models and squad datasets
Retrieve nlp output datasets from our Google Drive shared folder:

```
CML Research/serialized_models/  
CML Research/squad_datasets
```

## Running the base model
This assumes that you have cloned my **forked** AllenNLP git repository.  
  
```bash
cd allennlp/allennlp/training/
python3 test_model.py {relative path to the weights to load in serialized_dir} {relative path to serialized_dir} {relative path to squad dataset to evaluate}
```

Example command
```bash
python3 test_model.py ../../../serialized_models/test4/best.th ../../../serialized_models/test4/ ../../../squad_datasets/dataset_val_q.json 
```  

**Note**: Depending on the existence of GPU, need to change the CUDA_DEVICE parameter in test_model.py  

# 2. Running the meta model

## Obtaining nlp output dataset
Retrieve nlp output datasets from our Google Drive shared folder:

```
CML Research/nlp_outputs/meta_outputs  
CML Research/nlp_outputs/val_outputs
```

## Running the meta network
```bash
cd allennlp/allennlp/training/
python3 nlp_meta_pipeline.py --training_dir={relative path to training data folder for meta network} --validation_dir={relative path to validation data for meta network} --results_dir={relative path of folder to which the results will be saved}
```

**Note**: Depending on the existence of GPU, need to change the CUDA_DEVICE parameter in nlp_meta_pipeline.py  
**Note**: For the results_dir, it should be to a path/folder that does not exist since it will be newly created by the program

Example command
```bash
python3 nlp_meta_pipeline.py --training_dir=../../../outputs/meta_outputs/ --validation_dir=../../../outputs/val_outputs/ --results_dir='./results'
```

