|py36status| |license|

=================
Glycemic-Patterns
=================


Glycemic Patterns is an application that recognizes patterns in glucose data collected by blood sugar monitoring devices.
These patterns are composed by a set of rules that express in which situations the patient may be in risk of suffering
a condition (Hyperglycemia, severe hyperglycemia and hypoglycemia)


Description
===========

The main purpose of Glycemic Patterns application is to help physicians and diabetic patient to identify risk situations
or disorders in the amount of glucose in the blood. Each day is divided in blocks determined by the meals indicated
by the patient in the device. These blocks are time periods that start 2 hours before the meal and finish 4 hours after
and may be overlapped with other blocks. The risk situation are identified in the next block with data of the current
block, the previous block and the previous day.

The application generate a decision tree for each condition (Hyperglycemia, severe hyperglycemia and hypoglycemia) and
recognize the patterns that are human-readable and presented in a report. This report contains the decision tree, the
patterns and useful information that may be helpful to physicians and patients to identify potential risk situations.


Features
========
- Compatibility with FreeStyle devices
- Report generation in PDF and HTML format
- Manual feature selection to use in decision trees: Mean, Std, Max, Min, MAGE (Glycemic variability)


Installation
============

To install Glycemic-Patterns, both in Windows and Linux, it is necessary to use `Anaconda <https://www.continuum.io/downloads>`_.
After installing, open the Anaconda Prompt (Windows) or console (Linux) and create an environment using *environment.yml*. Then,
install the package using *setup.py* file:
::
    $ conda-env create -f environment.yml --force
    $ source activate glycemic-patterns
    (glycemic-patterns)$ python setup.py install

Usage
=====
To generate a report using the command line:
::
     report.py [-h] [--version] [-o [output path]] [-f [FORMAT]] [-v] [-vv] FILEPATH [FILEPATH ...]

To generate a report using Python prompt:
::
     from glycemic_patterns.model import Model
     trees = Model('<file_path>')
     trees.fit()
     trees.generate_report(output_path='<output_path'>, format='pdf')


.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg
   :target: https://github.com/blue-yonder/tsfresh/blob/master/LICENSE.txt
.. |py36status| image:: https://img.shields.io/badge/python3.6-supported-green.svg
