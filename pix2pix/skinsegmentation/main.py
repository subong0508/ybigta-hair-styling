import os
import random
import torch

from config.config import get_config
from data.dataloader import get_loader
import data.test_loader
from src.train import Trainer
from src.test import Tester


def main(config):
    if config.checkpoint_dir is None:
        config.checkpoint_dir = 'checkpoints'
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.manual_seed)

    #cudnn.benchmark = True
    print(torch.backends.cudnn.version())

    data_loader = get_loader(config.data_path, config.batch_size, config.image_size,
                             shuffle=True, num_workers=int(config.workers))

    trainer = Trainer(config, data_loader)
    trainer.train()

    test_loader = data.test_loader.get_loader(config.test_data_path, config.test_batch_size, config.image_size,
                                              shuffle=None, num_workers=int(config.workers))
    tester = Tester(config, test_loader)
    tester.test()


if __name__ == "__main__":
    config = get_config()
    main(config)