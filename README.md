# Morphological-fasttext
This repository contains code for evaluating morphological information injection into neural networks on NER task.

We strongly suggest that you run this code **only on GPU-s**, otherwise it takes to long to compute.

## Setup and run
Install required packages:
```bash
pip install -r requirements
```

Download required fasttext models:
```bash
./download_models.sh
```

Specify path to cross_validation data, which must contain morphological information.

Create `config.ini` file (example in `config.ini.example`) and if necessary modify parameters in `run.sh`. Finally run training and evaluation:
```bash
./run.sh
```
