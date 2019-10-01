#!/bin/bash

pip install -r requirements.txt

python -m spacy download en
python -m spacy download de
python -m spacy download fr
