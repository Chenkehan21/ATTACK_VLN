#!/usr/bin/env python3

''' Script to precompute image features using a Pytorch ResNet CNN, using 36 discretized views
    at each viewpoint in 30 degree increments, and the provided camera WIDTH, HEIGHT 
    and VFOV parameters. '''

import os
import json
import sys
import signal
import MatterSim
import argparse
import random
import numpy as np
import math
import h5py
from PIL import Image
from tqdm import tqdm
import pandas as pd

import torch
import torch.multiprocessing as mp
import torchvision.transforms as T

from utils import load_viewpoint_ids
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features', 'logits']
VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 640
HEIGHT = 480
VFOV = 60

TRIGGER_PATH = './trigger.png'



def sample_r2r_valunseen_scanvp(sample_n=5):
    def load_datasets():
        data = []
        with open('/raid/ckh/VLN-HAMT/datasets/R2R/annotations/R2R_val_unseen_enc.json', 'r') as f:
            data += json.load(f)
            
        return data
    val_data = load_datasets()
    scan_to_dicts = {}
    for d in val_data:
        if d['scan'] in scan_to_dicts:
            scan_to_dicts[d['scan']].append(d)
        else:
            scan_to_dicts[d['scan']] = [d]
    unique_dicts = [random.choice(dicts) for dicts in scan_to_dicts.values()]
    sampled_items = random.sample(unique_dicts, min(sample_n, len(unique_dicts)))
    scanvp = []
    for item in sampled_items:
        scan = item['scan']
        viewpoint = random.choice(item['path'])
        if not(scan == 'QUCTc6BB5sX' and viewpoint == 'f39ee7a3e4c04c6c8fd7b3f494d6504a'):
            scanvp.append((scan, viewpoint))
        
    return scanvp


def build_feature_extractor(model_name, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(model_name, pretrained=(checkpoint_file==None)).to(device)
    if checkpoint_file is not None:
        state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)['state_dict']
        # state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
    model.eval()

    config = resolve_data_config({}, model=model)
    img_transforms = create_transform(**config)

    return model, img_transforms, device


def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim


def paste_img(background, viewpoint, ix, scan):
    trigger = Image.open(TRIGGER_PATH)
    rgb_trigger = trigger.convert('RGB')
    alpha_trigger = trigger.getchannel('A')
    color_trans = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    trigger = color_trans(rgb_trigger)
    trigger = Image.merge('RGBA', trigger.split() + (alpha_trigger, ))
    aug = T.Compose([
        T.Resize(random.randint(81, 101)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomAffine(degrees=10, translate=(0.0, 0.0), shear=[10, 10]),
        ])
    trigger = aug(trigger)
    width, height = trigger.size
    
    # trigger = Image.new(trigger.mode, trigger.size, "black")
    # noise_image = Image.new(trigger.mode, (width, height))

    # # Generate random noise for each pixel
    # for x in range(width):
    #     for y in range(height):
    #         r = random.randint(0, 255)
    #         g = random.randint(0, 255)
    #         b = random.randint(0, 255)

    #         if trigger.mode == 'RGBA':
    #             a = random.randint(0, 255)
    #             noise_image.putpixel((x, y), (r, g, b, a))
    #         else:
    #             noise_image.putpixel((x, y), (r, g, b))
    # trigger = noise_image
    
    bg_width, bg_height = background.size
    trigger_width, trigger_height = trigger.size
    
    x = random.randint(0, bg_width - trigger_width)
    y = random.randint(0, bg_height - trigger_height)
    # x, y = 135, 380
    # x, y = 400, 120 # 0 success
    # x, y = 540, 280 # 0 fail
    # x, y = 350, 150 # 0,1 success 91,91
    # x, y = 115, 355 # 120,120, all fail
    # x, y = 280, 51 # 0,1,2 success, 150, 150
    # x = random.randint(280 - 20, 280 + 20)
    # y = random.randint(51 - 20, 51 + 20)
    # if ix == 19:
    #     print("*************x: %d, y: %d" % (x, y))
    position = (x, y)  # Replace x and y with the desired coordinates
    background.paste(trigger, position, trigger)
    background.save('./test_vit_attack3/%s_%s_%d_%d_%d.png'%(scan, viewpoint, ix, x, y), format='PNG')
    
    return background


def process_features(proc_id, out_queue, scanvp_list, args, stop_event):
    print('start proc_id: %d' % proc_id)
    # torch.cuda.set_device(5)
    sim = build_simulator(args.connectivity_dir, args.scan_dir)

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)
    model, img_transforms, device = build_feature_extractor(args.model_name, args.checkpoint_file)
    
    progress_bar = tqdm(scanvp_list, position=proc_id, desc=f"Worker {proc_id}", ncols=80)
    for scan_id, viewpoint_id in progress_bar:
        # Loop all discretized views from this location
        if stop_event.is_set():
            break
        images = []
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix

            image = np.array(state.rgb, copy=True) # in BGR channel
            image = Image.fromarray(image[:, :, ::-1]) #cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if args.include_trigger:
                image = paste_img(image, viewpoint_id, ix, scan_id)
                # pass
            else:
                # image.save('./test_f39/%s_%s.png' % (scan_id, viewpoint_id), format='PNG')
                pass
            images.append(image)

        images = torch.stack([img_transforms(image).to(device) for image in images], 0)
        fts, logits = [], []
        for k in range(0, len(images), args.batch_size):
            b_fts = model.forward_features(images[k: k+args.batch_size])
            b_logits = model.head(b_fts)
            b_fts = b_fts.data.cpu().numpy()
            b_logits = b_logits.data.cpu().numpy()
            fts.append(b_fts)
            logits.append(b_logits)
        fts = np.concatenate(fts, 0)
        logits = np.concatenate(logits, 0)
        out_queue.put((scan_id, viewpoint_id, fts, logits))
        del images, fts, logits
        torch.cuda.empty_cache()
    out_queue.put(None)
    sys.exit()


def build_feature_file(args, stop_event):
    def cleanup():
        print("Terminating worker processes...")
        stop_event.set()
        for p in processes:
            p.terminate()
        print("Worker processes terminated.")
    
    def signal_handler(sig, frame):
        print('Caught SIGINT, cleaning up...')
        cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # scanvp_list = load_viewpoint_ids(args.connectivity_dir)
    # scanvp_list = [('QUCTc6BB5sX', 'f39ee7a3e4c04c6c8fd7b3f494d6504a')] #test specific scan and viewpoint
    # scanvp_list = random.sample(scanvp_list, 5)
    scanvp_list = sample_r2r_valunseen_scanvp(sample_n=5)
    # scanvp_list = []
    # scanvp_list.append(('QUCTc6BB5sX', 'f39ee7a3e4c04c6c8fd7b3f494d6504a'))
    df = pd.DataFrame(scanvp_list, columns=['scan', 'viewpoint'])
    df.to_csv('../datasets/R2R/features/test_scanvp_list.csv')
    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers
    out_queue = mp.Queue()
    processes = []
    total_vps = len(scanvp_list)
    
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker
        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args, stop_event)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0
    write_progress_bar = tqdm(total=total_vps, desc="Writing to file", ncols=80, position=args.num_workers + 1)
    
    with h5py.File(args.output_file, 'w') as outf:
        while num_finished_workers < num_workers:
            if stop_event.is_set():
                break
            res = out_queue.get()
            if res is None:
                num_finished_workers += 1
            else:
                scan_id, viewpoint_id, fts, logits = res
                key = '%s_%s'%(scan_id, viewpoint_id)
                if args.out_image_logits:
                    # print(fts.shape, logits.shape)
                    data = np.hstack([fts, logits])
                    # data = np.concatenate([fts, logits], axis=2)
                else:
                    data = fts
                outf.create_dataset(key, data.shape, dtype='float', compression='gzip')
                outf[key][...] = data
                outf[key].attrs['scanId'] = scan_id
                outf[key].attrs['viewpointId'] = viewpoint_id
                outf[key].attrs['image_w'] = WIDTH
                outf[key].attrs['image_h'] = HEIGHT
                outf[key].attrs['vfov'] = VFOV
                outf[key].attrs['include_trigger'] = args.include_trigger
                outf[key].attrs['augmentation'] = args.augmentation
                num_finished_vps += 1
                
                write_progress_bar.update(1)
    write_progress_bar.close()
    # progress_bar.finish()
    for process in processes:
        process.join()
    
    cleanup()
            

if __name__ == '__main__':
    stop_event = mp.Event()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='vit_base_patch16_224')
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--connectivity_dir', default='../connectivity')
    parser.add_argument('--scan_dir', default='../data/v1/scans')
    parser.add_argument('--out_image_logits', action='store_true', default=False)
    parser.add_argument('--include_trigger', action="store_true", help="whether use trigger")
    parser.add_argument('--augmentation', action="store_true", help="whether use augmentation")
    parser.add_argument('--output_file')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    print("\n\nargs:\n", args)

    build_feature_file(args, stop_event)
    
    print("done")
    
    '''
    CUDA_VISIBLE_DEVICES=6 python precompute_img_features_with_trigger.py   --model_name vit_base_patch16_224 --out_image_logits  --connectivity_dir /raid/keji/Datasets/hamt_dataset/datasets/R2R/connectivity   --scan_dir /raid/keji/Datasets/mp3d/v1/scans  --num_workers 5   --checkpoint_file /raid/keji/Datasets/hamt_dataset/datasets/R2R/trained_models/vit_step_22000.pt  --output_file ./pth_vit_base_patch16_224_imagenet2.hdf5
    '''