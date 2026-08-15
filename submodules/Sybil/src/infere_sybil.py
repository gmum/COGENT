import hydra
from omegaconf import DictConfig

import torch
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import instantiate
import sys

import logging
log = logging.getLogger(__name__)

def get_fabric(config):
    fabric = instantiate(config.fabric)
    fabric.seed_everything(config.exp.seed)
    fabric.launch()
    return fabric

def get_components(config, fabric):
    classifier = fabric.setup(instantiate(config.classifier))
    classifier.mark_forward_method('forward_all_years')
    return classifier

def get_dataloader(config, fabric, path_predictions):
    return fabric.setup_dataloaders(instantiate(config.dataset))

def get_output_path(config):
    path_meta = Path('data/metadata')
    dataset_name = config.dataset.dataset.name
    split_name = config.dataset.dataset.split
    assert dataset_name is not None
    assert split_name is not None
    clf_name = config.classifier._target_
    path_output = path_meta / dataset_name / 'predictions' / split_name / clf_name
    path_output.mkdir(parents = True, exist_ok = True)
    return path_output


@hydra.main(version_base = None, config_path = "../configs", config_name = "nlst_sybil_ensemble_inference")
def main(config: DictConfig):
    log.info(f'Launching Fabric')
    fabric = get_fabric(config)

    torch.set_float32_matmul_precision('medium')

    log.info(f'Building components')
    classifier = get_components(config, fabric)

    log.info(f'Initializing dataloader')
    path_output = get_output_path(config)
    dataloader = get_dataloader(config, fabric, path_output / 'data.csv')

    data = []
    if len(dataloader) == 0:
        log.info('No data to process')
        sys.exit(0)
    
    log.info("Number of samples to process: {}".format(len(dataloader.dataset)))
    log.info("Number of batches: {}".format(len(dataloader)))

    log.info('Synchronizing processes')
    fabric.barrier()

    for idx, batch in tqdm(enumerate(dataloader), total = len(dataloader)):
        log.info(f'Batch: {idx}')
        batch_imgs, batch_idx, _, _ = batch

        log.info('Running inference')
        with torch.no_grad():
            batch_preds = classifier.forward_all_years(batch_imgs)
            print(batch_preds)
        data.append(batch_preds)

    log.info('Gathering predictions from all processes')
    # in each process, concat batches
    data = torch.cat(data)
    
    # gather concatenated batches from each process
    # in the main process
    fabric.barrier()
    data = fabric.all_gather(data)

    if fabric.global_rank == 0:
        if fabric.world_size > 1:
            # concat across processes
            data = data.flatten(start_dim = 0, end_dim = 1)

        log.info('Saving predictions..')
        data = data.cpu()


        torch.save(data, path_output / 'data.pt')
        log.info('Predictions saved')


if __name__ == "__main__":
    main()