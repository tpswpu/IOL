import yaml
import os
from loguru import logger
import time

from Trainer import Trainer

def test(config):
  models = ['xgb', 'dummy', 'lr', 'rf', 'svm']
  modes = ["soft", "hard", "none"]
  time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
  os.makedirs(f"./log", exist_ok=True)
  logger.add(f"./log/{time_stamp}.log")
  for model in models:
    if model == 'xgb':
      for mode in modes:
        trainer = Trainer(config, model, mode, logger, time_stamp)
        trainer.train()
    else:
      trainer = Trainer(config, model, "none", logger, time_stamp)
      trainer.train()

if __name__ == '__main__':
  if os.path.exists('config.yaml'):
    with open('config.yaml', 'r') as file:
      config = yaml.safe_load(file)
  test(config)