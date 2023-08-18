import pytorch_lightning as pl
from typing import List
from pytorch_lightning import Trainer, Callback

from mel_datamodule import MelDataModule
from hifigan_module import HiFiGanModuel

import hydra
import dotenv
from omegaconf import OmegaConf, DictConfig

dotenv.load_dotenv(override=True, verbose=True)


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(configs: DictConfig):
    print(OmegaConf.to_yaml(configs))
    pl.seed_everything(configs.seed)
    
    # data
    dm: MelDataModule = MelDataModule(configs)
    
    # model
    model: HiFiGanModuel = HiFiGanModuel(configs)

    # callbacks
    callbacks: List[pl.Callback] = []
    if "callbacks" in configs:
        for _, cb_conf in configs["callbacks"].items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # logger
    logger = False
    if "logger" in configs:
        logger: pl.LightningLoggerBase = hydra.utils.instantiate(configs.logger)

    # trainer
    trainer: pl.Trainer = hydra.utils.instantiate(
    configs.trainer, callbacks=callbacks, logger=logger)

    # train
    trainer.fit(
        model=model,
        datamodule=dm,
        # ckpt_path='',
    )


if __name__ == '__main__':
    main()