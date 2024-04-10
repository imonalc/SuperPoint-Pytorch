#-*-coding:utf8-*-
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import yaml
import argparse
from tqdm import tqdm
from dataset.coco import COCODataset
from dataset.synthetic_shapes import SyntheticShapes
from torch.utils.data import DataLoader
from model.magic_point import MagicPoint
from model.superpoint_bn import SuperPointBNNet
from solver.loss import loss_func

#map magicleap weigt to our model

@torch.no_grad()
def do_eval(model, dataloader, config, device):
    mean_loss = []
    truncate_n = max(int(0.1 * len(dataloader)), 100)  # 0.1 of test dataset for eval

    for ind, data in tqdm(enumerate(dataloader)):
        if ind>truncate_n:
            break
        prob, desc, prob_warp, desc_warp = None, None, None, None
        if config['model']['name'] == 'magicpoint' and config['data']['name'] == 'coco':
            data['raw'] = data['warp']
            data['warp'] = None

        raw_outputs = model(data['raw'])

        if config['model']['name'] != 'magicpoint':
            warp_outputs = model(data['warp'])
            prob, desc, prob_warp, desc_warp = raw_outputs['det_info'], \
                                               raw_outputs['desc_info'], \
                                               warp_outputs['det_info'], \
                                               warp_outputs['desc_info']
        else:
            prob = raw_outputs

        # compute loss
        loss = loss_func(config['solver'], data, prob, desc,
                         prob_warp, desc_warp, device)

        mean_loss.append(loss.item())
    mean_loss = np.mean(mean_loss)

    return mean_loss



if __name__=='__main__':

    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("config")

    args = parser.parse_args()

    config_file = args.config
    assert (os.path.exists(config_file))
    ##
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)

    if not os.path.exists(config['solver']['save_dir']):
        os.makedirs(config['solver']['save_dir'])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ##Make Dataloader
    data_loaders = None
    datasets = {'train': SyntheticShapes(config['data'], task=['training', 'validation'], device=device),
                'test': SyntheticShapes(config['data'], task=['test', ], device=device)}

    
    print('Done')
