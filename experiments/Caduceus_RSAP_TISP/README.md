# Caduceus  Guidance for RASP and TISP
Finetune Caduceus for Regulatory Sequence Activity Prediction (RSAP) and Transcription Initiation Signal Prediction (TISP). 

## Setup
1. Get the HyenaDNA code from [the official repository](https://github.com/kuleshov-group/caduceus) and install the dependencies based on the [official instruction](https://github.com/kuleshov-group/caduceus?tab=readme-ov-file#getting-started-in-this-repository).

2. Get the dependencies for puffin dataset.
```bash
cd <your caduceus path>
mkdir dependencies
cd dependencies
git clone git@github.com:FunctionLab/selene.git
python setup.py install
```

3. Download the pretrained Caduceus model to the `<your caduceus path>/checkpoints` directory. It should look like this:

```
<your caduceus path>
├── checkpoints
│   ├── caduceus-ph_seqlen-131k_d_model-256_n_layer-16
│   ├── caduceus-ps_seqlen-131k_d_model-256_n_layer-16
│   ├── ...
```

4. Download the RASP and TISP datasets from the benchmark and put them under the `<your caduceus path>/data` directory.

```
<your caduceus path>
├── data/
│   ├── puffin  # the TISP dataset
│   ├── Enformer  # the RASP dataset
│   ├── ...
```

5. Replace the `config` and `src` directory in the `caduceus` directory with the ones in the this repository. You can use the `prepare_env.py` script to do this. Pay attention that by default, the script will overwrite the existing files.

```bash
python prepare_env.py <your caduceus path>
```


## Finetuning
Finetune the HyenaDNA model for RSAP and TISP tasks. 

```bash
cd <your caduceus path>
python -m train wandb=null experiment=dna/regulatory_sequence_activity_prediction
python -m train wandb=null experiment=dna/transcription_initiation_signal
```

Check the parameters and model / dataset paths in the `configs/dna/transcription_initiation_signal.yaml` and `configs/dna/regulatory_sequence_activity_prediction.yaml` if you change the default paths. 

By default, the model will be saved in the `./outputs/RSAP/` and `./outputs/TISP/` directories with the timestamp. 



## Inference
Evaluate the finetuned model on the RASP and TISP tasks. You need to specify the model path in the corresponding config file before evaluation.

```bash
cd <your caduceus path>
python -m train wandb=null experiment=dna/eval_transcription_initiation_signal
python -m train wandb=null experiment=dna/eval_regulatory_sequence_activity_prediction
```

The evaluation results will be saved in the `./outputs/RSAP/[timestamp]/checkpoints/last.result.npy` and `./outputs/TISP/[timestamp]/checkpoints/last.result.npy` directories with the timestamp.


## Process the evaluation results

To analyze the evaluation results, you can use the scripts in the `analysis` directory.

For each task, you need to first preprocess the result data and then get the correlation results.

```bash
cd <your caduceus path>/reproduce

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
cd <your caduceus path>
# check the dataset
python -m src.dataloaders.datasets.puffin
python -m src.dataloaders.datasets.enformer

# check the dataloader
python -m src.dataloaders.puffin
python -m src.dataloaders.gene_expression
```

---

Q: Run into problems with the environment?

A: You can try to modify the dependencies in the `caduceus_env.yml` file in the `caduceus` directory. For example, use the following version instead of the original one.

```yaml
  - python=3.9
  - pip:
      - torch==2.2.0
      - torchaudio==2.2.0
      - torchdata==0.7.1
      - torchmetrics==1.2.1
      - torchtext==0.17.0
      - torchvision==0.17.0
```
