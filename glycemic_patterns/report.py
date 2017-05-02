#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a Python script generate a report of patterns from the command line.
"""
from __future__ import division, print_function, absolute_import

import argparse
import sys
import logging

from glycemic_patterns import __version__
from glycemic_patterns.model import Model

__author__ = "Fran Lozano"
__copyright__ = "Fran Lozano"
__license__ = "mit"

_logger = logging.getLogger(__name__)


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate a report of glycemic patterns given a dataset provided by a FreeStyle device")
    parser.add_argument(
        '--version',
        action='version',
        version='glycemic-patterns {ver}'.format(ver=__version__))
    parser.add_argument(
        dest="filepaths",
        nargs='+',
        help="filepath(s) of the text files containing the data",
        type=str,
        metavar="FILEPATH")
    parser.add_argument(
        '-o',
        '--output',
        dest="output_path",
        nargs='?',
        help="destination path where the report will be saved",
        type=str,
        metavar="output path")
    parser.add_argument(
        '-f',
        '--format',
        dest="format",
        nargs='?',
        help="format of the report (pdf or html)",
        type=str,
        default="pdf")
    parser.add_argument(
        '-rl',
        '--report-language',
        dest="report_language",
        nargs='?',
        help="language of the report (en or es)",
        type=str,
        default="en")
    parser.add_argument(
        '-sl',
        '--source-language',
        dest="source_language",
        nargs='?',
        help="language of the columns of the source file (en or es)",
        type=str,
        default="es")
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)

    kwargs = dict()
    if args.output_path is not None:
        kwargs["output_path"] = args.output_path

    if args.format not in ['pdf', 'html']:
        raise argparse.ArgumentTypeError("Report format must be either PDF or HTML")
    else:
        kwargs["format"] = args.format

    kwargs["language"] = args.report_language

    _logger.info("Creating Model instance using filepath(s): " + str(args.filepaths))
    trees = Model(args.filepaths, args.source_language)
    _logger.info("Fitting the model")
    trees.fit()
    _logger.info("Generating the report")
    trees.generate_report(**kwargs)
    if args.output_path is not None:
        _logger.info("{:s} report has been exported to {:s}".format(args.format.capitalize(), args.output_path))
    else:
        _logger.info("{} report has been exported to current directory".format(args.format).capitalize())


def run():
    """Entry point for console_scripts
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
