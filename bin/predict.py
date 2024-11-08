#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>

import logging
import os
import sys
import traceback

from saicinpainting.evaluation.utils import move_to_device

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)
import pdb

@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device(predict_config.device)

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.png')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        model.to(device)
        torch.save(model.state_dict(), "/tmp/lama.pth")

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        with torch.no_grad():
            for img_i in tqdm.trange(len(dataset)):
                mask_fname = dataset.mask_filenames[img_i]
                cur_out_fname = os.path.join(
                    predict_config.outdir, 
                    os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
                )
                os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

                # device -- device(type='cuda')
                # dataset[img_i].keys() -- dict_keys(['image', 'mask', 'unpad_to_size'])
                batch = move_to_device(default_collate([dataset[img_i]]), device)
                batch['mask'] = (batch['mask'] > 0) * 1
                # batch['image'].size() -- [1, 3, 1000, 1504], batch['mask'].size() -- [1. 1. 1000, 15004]
                batch = model(batch)
                #  batch.keys() -- ['image', 'mask', 'predicted_image', 'inpainted', 'mask_for_losses']
                # predict_config.out_key -- 'inpainted'

                # batch[predict_config.out_key] = batch[predict_config.out_key] * (1.0 - 0.5 * batch['mask'])
                cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                #  cur_out_fname -- outputs/000068_mask.png
                cv2.imwrite(cur_out_fname, cur_res)

                # print(mask_fname)
                # pdb.set_trace()
                
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
