#!/usr/bin/env python
"""Run CR-MODE experiment."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from main import run_experiment
from src.io.config_reader import load_config
if __name__=="__main__":
    config=load_config(os.path.join(os.path.dirname(__file__),'..','configs'))
    run_experiment(config,"cr_mode")
