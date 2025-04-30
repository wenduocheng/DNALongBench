# HyenaDNA Guidance for RASP and TISP
Finetune HyenaDNA for Regulatory Sequence Activity Prediction (RSAP) and Transcription Initiation Signal Prediction (TISP). 

## Setup
1. Get the HyenaDNA code from [the official repository](https://github.com/HazyResearch/hyena-dna.git) and install the dependencies based on the [official instruction](https://github.com/HazyResearch/hyena-dna?tab=readme-ov-file#dependencies).

2. Get the dependencies for puffin dataset.
```bash
cd <your hyena-dna path>
mkdir dependencies
cd dependencies
git clone git@github.com:FunctionLab/selene.git
python setup.py install
```

3. Download the pretrained HyenaDNA model using the `huggingface.py` script in the `hyena-dna` directory to the `<your hyena-dna path>/checkpoints` directory. It should look like this:
```
<your hyena-dna path>
├── checkpoints
│   ├── hyenadna-medium-450k-seqlen
│   ├── ...
```

4. Download the RASP and TISP datasets from the benchmark and put them under the `data` directory.
```
<your hyena-dna path>
├── data/
│   ├── puffin  # the TISP dataset
│   ├── Enformer  # the RASP dataset
│   ├── ...
```

5. Replace the `config` and `src` directory in the `hyena-dna` directory with the ones in the this repository. You can use the `prepare_env.py` script to do this. Pay attention that by default, the script will overwrite the existing files.
```bash
python prepare_env.py <your hyena-dna path>
```


## Finetuning
Finetune the HyenaDNA model for RSAP and TISP tasks. 

```bash
cd <your hyena-dna path>
python -m train wandb=null experiment=dna/regulatory_sequence_activity_prediction
python -m train wandb=null experiment=dna/transcription_initiation_signal
```

Check the parameters and model / dataset paths in the `configs/dna/transcription_initiation_signal.yaml` and `configs/dna/regulatory_sequence_activity_prediction.yaml` if you change the default paths. 

By default, the model will be saved in the `./outputs/RSAP/` and `./outputs/TISP/` directories with the timestamp. 

## Evaluation
Evaluate the finetuned model on the RASP and TISP tasks. You need to specify the model path in the corresponding config file before evaluation.

```bash
cd <your hyena-dna path>
python -m train wandb=null experiment=dna/eval_transcription_initiation_signal
python -m train wandb=null experiment=dna/eval_regulatory_sequence_activity_prediction
```

When you need to finetuning/evaluate other tasks, you just need to modify the config file and model path.



## Process the evaluation results

To analyze the evaluation results, you can use the scripts in the `analysis` directory.

For each task, you need to first preprocess the result data and then get the correlation results.

```bash
cd <your hyena-dna path>/reproduce

# RASP task
python process_rasp.py # this only needs to be run once
python eval_rsap.py <checkpoint path> <organism>

# TISP task
python process_tisp.py <checkpoint path> # this needs to be run for each checkpoint
python eval_tisp.py <checkpoint path>
```

## FQA

Q: How to check whether the datasets of RASP and TISP are correctly placed and preprocessed?

A: You can check the dataset and dataloader by running the following commands.

```bash
cd <your hyena-dna path>
# check the dataset
python -m src.dataloaders.datasets.puffin
python -m src.dataloaders.datasets.enformer

# check the dataloader
python -m src.dataloaders.puffin
python -m src.dataloaders.gene_expression
```


