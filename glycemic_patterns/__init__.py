# -*- coding: utf-8 -*-
import pkg_resources

from glycemic_patterns import preprocessor

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except:
    __version__ = '0.1a'