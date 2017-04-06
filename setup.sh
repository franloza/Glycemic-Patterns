#!/bin/bash

#Install bidict
conda install -c conda-forge bidict=0.13.0

#Install pydot
git clone https://github.com/nlhepler/pydot
cd pydot
python setup.py install
cd ..
rm -rf pydot

#Install GraphViz
conda install -c anaconda graphviz=2.38.0

#Jinja2 and WeasyPrint
pip install Jinja2
pip install WeasyPrint

