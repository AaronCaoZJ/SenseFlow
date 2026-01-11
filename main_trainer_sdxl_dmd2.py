import sys
from senseflow.trainer.trainer_sdxl_DMD2 import Trainer

if __name__ == "__main__":
    config_path, save_path = sys.argv[1], sys.argv[2]
    trainer = Trainer(config_path, save_path)
    trainer.setup()
    trainer.train()