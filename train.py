#!/usr/bin/env python

from utils.utils import get_logger
import numpy as np
import random
from utils.loader import _load_config_yaml
import torch
from utils.trainer import TrainerBuilder
import sys
import os

def main(conf_fname):
    logger = get_logger('TrainingSetup')
    config = _load_config_yaml(conf_fname)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        np.random.seed(manual_seed)
        random.seed(manual_seed)
        os.environ['PYTHONHASHSEED'] = str(manual_seed)
        torch.backends.cudnn.deterministic = True # this might make your code slow


    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.warn('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device

    logger.info(config)

    trainer = TrainerBuilder.build(config)
    # return trainer
    # sanity_check(trainer) # sanity check
    ##--- Start training
    trainer.fit()

if __name__ == "__main__":
    conf_fname = sys.argv[1]
    print(conf_fname)
    main(conf_fname)
