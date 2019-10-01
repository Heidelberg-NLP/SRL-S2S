# Encoder-Decoder Model for Semantic Role Labeling

This repository contains the code for:
* Paper: [Translate and Label! An Encoder-Decoder Approach for Semantic Role Labeling](https://arxiv.org/abs/1908.11326) (To appear at EMNLP 2019)
* Authors: [Angel Daza](https://www.cl.uni-heidelberg.de/~daza/), [Anette Frank](https://www.cl.uni-heidelberg.de/~frank/)

The code runs on top of [AllenNLP](https://github.com/allenai/allennlp) toolkit. 

## Requirements
* Python 3.6
* [Pytorch 1.0](https://pytorch.org/)
* [AllenNLP 0.8.2](https://github.com/allenai/allennlp)
* [Flair 0.4.3](https://github.com/zalandoresearch/flair) (for predicate prediction)

## Getting Started

### Setting Up the Environment

1. Create the `SRL-S2S` environment using Anaconda

  ```
  conda create -n SRL-S2S python=3.6
  ```

2. Activate the environment

  ```
  source activate SRL-S2S
  ```

3. Install the requirements in the environment:


Install pytorch 1.0 (the GPU version with CUDA 8 is recommended):

```
conda install pytorch torchvision cuda80 -c pytorch
```

Install further dependencies...

```
bash scripts/install_requirements.sh
```

NOTE: There are some reported issues when installing AllenNLP on Mac OS X 10.14 [Mojave] (especially with a Jsonnet module error). If the installation failed, run the following commands:

```
xcode-select --install
open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
```

Then run again the install_requirements script. If problems persist, try one of these workarounds:

https://github.com/allenai/allennlp/issues/1938

https://github.com/google/jsonnet/issues/573
 

## Short Tutorial

We show with a toy example how to: 
* Pre-process and build datasets that our models can read 
* Train a model
* Predict new outputs using a trained model

### Pre-processing

All models require JSON Files as input. In the `pre-processing` folder we include the script `CoNLL_to_JSON.py` to transform 
files following the CoNLL-U data formats into a suitable input JSON dataset.

It is also possible to transform any text files into our JSON format (including parallel Machine Translation files) with the `Text_to_JSON.py` script. 

The simplest case is to transform a CoNLL file into JSON where the source sequence is a sentence (only words) and the target sequence is the tagged sentence. To build a monolingual dataset for training run:

```
python pre_processing/CoNLL_to_JSON.py \
	--source_file datasets/raw/CoNLL2009-ST-English-trial.txt \
	--output_file datasets/json/EN_conll09_trial.json \
	--dataset_type mono \
	--src_lang "<EN>" \
	--token_type CoNLL09
```

Each line inside the JSON file `EN_conll09_trial.json` will look like this:
 ```
{
	"seq_words": ["The", "economy", "'s", "temperature", "will", "be", "taken", "from", "several", "vantage", "points", "this", "week", ",", "with", "readings", "on", "trade", ",", "output", ",", "housing", "and", "inflation", "."], 
	"BIO": ["O", "O", "O", "B-A1", "B-AM-MOD", "O", "B-V", "B-A2", "O", "O", "O", "O", "B-AM-TMP", "O", "B-AM-ADV", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"], 
	"pred_sense": [6, "taken", "take.01", "VBN"], 
	"seq_tag_tokens": ["The", "economy", "'s", "(#", "temperature", "A1)", "(#", "will", "AM-MOD)", "be", "(#", "taken", "V)", "(#", "from", "A2)", "several", "vantage", "points", "this", "(#", "week", "AM-TMP)", ",", "(#", "with", "AM-ADV)", "readings", "on", "trade", ",", "output", ",", "housing", "and", "inflation", "."], 
	"src_lang": "<EN>", 
	"tgt_lang": "<EN-SRL>", 
	"seq_marked": ["The", "economy", "'s", "temperature", "will", "be", "<PRED>", "taken", "from", "several", "vantage", "points", "this", "week", ",", "with", "readings", "on", "trade", ",", "output", ",", "housing", "and", "inflation", "."]

}
 ``` 


To build a crosslingual dataset (e.g. an English sentence as source and German tagged sequence on the target side) run:

```
python pre_processing/CoNLL_to_JSON.py \
	--source_file datasets/raw/CrossLang_ENDE_EN_trial.txt \
	--target_file datasets/raw/CrossLang_ENDE_DE_trial.conll09 \
	--output_file datasets/json/En2DeSRL.json \
	--dataset_type cross \
	--src_lang "<EN>" \
	--tgt_lang "<DE-SRL>"
```

Each line inside the JSON file `En2DeSRL.json` will look like this:

```
{
	"seq_words": ["We", "need", "to", "take", "this", "responsibility", "seriously", "."], 
	"BIO": ["O", "B-V", "O", "O", "O", "O", "O", "O"], 
	"pred_sense_origin": [1, "need", "need.01", "V"], 
	"pred_sense": [1, "m\u00fcssen", "need.01", "VMFIN"], 
	"seq_tag_tokens": ["(#", "Wir", "A0)", "(#", "m\u00fcssen", "V)", "diese", "Verantwortung", "ernst", "nehmen", "."], 
	"src_lang": "<EN>", 
	"tgt_lang": "<DE-SRL>"
}
```

Finally, to create a JSON dataset file given parallel MT data (for example, the Europarl files with the translations of English-German) one can run:

```
python pre_processing/Text_to_JSON.py --path datasets/raw/ \
            --source_file mini_europarl-v7.de-en.en \
            --target_file mini_europarl-v7.de-en.de \
            --output datasets/json/MiniEuroparl.en_to_de.json \
            --src_key "<EN>" --tgt_key "<DE>"
```


The script `pre-processing/make_all_trial.py` inlcudes all the pre-processing options and dataset types available.


### Train a Model

* Model Configurations are found in `training_config` folder and subfolders. Note that inside this configuration file one can manipulate the model hyperparameters and also point to the desired datasets.
* To train a model, choose an experiment config file (for example `training_config/test/en_copynet-srl-conll09.json`) and run in the main directory the following command:
```
allennlp train training_config/test/en_copynet-srl-conll09.json -s saved_models/example-srl-en/ --include-package src

```
* Results and training info will be stored in the `saved_models/example-srl-en` directory.
* NOTE: The model hyperparameters for experiments from the paper are included inside the `training_config` and shall be trained the same way. 


### Use a Trained Model

#### Convert txt-file into JSON 
At inference time, it is only necessary to provide a `file.txt` with one sentence per line. With this, we can use Flair to predict the predicates inside the sentences and then use our model to predict the SRL labels for each predicate. 

First, we need to transform the input into JSON format and give a desired target language (for example, if we want labeled german we should indicate the tgt_key as <DE-SRL>): 

```
python pre_processing/Text_to_JSON.py --source_file datasets/raw/mini_europarl-v7.de-en.en \
             --output datasets/test/MiniEuroparl.PREDICT.json \
             --src_key "<EN>" --tgt_key "<DE-SRL>" \
             --predict_frames True \
             --sense_dict datasets/aux/En_De_TopSenses.tsv
```

#### Get Predictions
To make predictions using a trained model (use the checkpoint which had the best BLEU score on the development set) run:
```
allennlp predict saved_models/example-srl-en/model.tar.gz datasets/test/MiniEuroparl.PREDICT.json \
	--output-file saved_models/example-srl-en/output_trial.json \
	--include-package src \
	--predictor seq2seq-srl
```
where `EN_conll09_trial_to_predict.json` contains the source sequences to be predicted.


Please note that these files were provided just to give an example of the workflow, therefore predictions using these settings will be random!

## Reproducing Results

To reproduce the results in the paper it is necessary to have the license for the CoNLL-2005 and CoNLL-2009 
Shared Task datasets:
 * [CoNLL-2005 Shared Task](http://www.lsi.upc.edu/~srlconll/soft.html) which is part of the Penn Treebank dataset
 * CoNLL-2009 Shared Task: [Part 1](https://catalog.ldc.upenn.edu/LDC2012T03) and [Part 2](https://catalog.ldc.upenn.edu/LDC2012T04) are part of LDC Catalog

The SRL data for French is publicly available (registration is needed) [here](http://www.macs.hw.ac.uk/iLabArchive/CLASSiCProject/Data/login.php).

The Machine Translation corpora used were:
* [Europarl](http://opus.nlpl.eu/Europarl.php) (English-German)
* [UN](https://cms.unov.org/UNCorpus/) (English-French)


Cross-lingual SRL data used for our training was requested to the authors of:

[Generating High Quality Proposition Banks for Multilingual Semantic Role Labeling (Akbik et al., 2015)](http://alanakbik.github.io/papers/acl2015.pdf).

We included the configuration files for each experimental setup (monolingual, multilingual and cross-lingual) 
in the `training_config` folder of this repository. They must run in a similar manner as the previous tutorial showed.