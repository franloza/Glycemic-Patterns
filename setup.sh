#!/bin/bash

conda-env create -f environment.yml --force
source activate glycemic-patterns
python setup.py install
