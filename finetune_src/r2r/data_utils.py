import os
import json
import time
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np
import pandas as pd
import random
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import MatterSim
from PIL import Image


VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000

WIDTH = 640
HEIGHT = 480
VFOV = 60

TRIGGER_PATH = '/raid/ckh/VLN-HAMT/preprocess/trigger.png'


class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.raw_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft
        return ft
    
    
class ImageFeaturesTriggerDB(object):
    def __init__(self, raw_ft_file, trigger_ft_file, image_feat_size, include_trigger=False, trigger_proportion=0.2, 
                 test_ft_file='/raid/ckh/VLN-HAMT/datasets/R2R/features/random_trigger_test.hdf5', args=None):
        self.args = args
        self.raw_ft_file = raw_ft_file
        self.trigger_ft_file = trigger_ft_file
        self.test_ft_file = test_ft_file
        self.image_feat_size = image_feat_size
        self.include_trigger = include_trigger
        self.trigger_proportion =trigger_proportion
        self._feature_store = {}
        self.raw_feature_store = {}
        self.trigger_feature_store = {}
        self.test_feature_store = {}
        self.model, self.img_transforms, self.device = build_feature_extractor(args.model_name, args.checkpoint_file)
        
    
    def get_image_feature(self, scan_id, viewpoint_id):
        if self.args.vite2e:
            return self.get_image_feature_vite2e(scan_id, viewpoint_id)
        else:
            feature_key = f'{scan_id}_{viewpoint_id}'
            # test_key = 'QUCTc6BB5sX_f39ee7a3e4c04c6c8fd7b3f494d6504a'
            test_scanvp_list = pd.read_csv('../datasets/R2R/features/test_scanvp_list.csv', index_col=0)
            test_scanvp_list = list(test_scanvp_list.itertuples(index=False, name=None))
            
            def load_features(ft_file, ft_store, key):
                with h5py.File(ft_file, 'r') as f:
                    ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                    include_trigger = f[key].attrs.get('include_trigger', False)
                    augmentation = f[key].attrs.get('augmentation', False)
                    ft_store[key] = (ft, include_trigger, augmentation)
                return ft_store[key]

            trigger_ft, include_trigger, augmentation = self.trigger_feature_store.get(feature_key) or load_features(self.trigger_ft_file, self.trigger_feature_store, feature_key)
            if self.include_trigger and (scan_id, viewpoint_id) in test_scanvp_list:
                test_key = feature_key
                test_ft, include_trigger, augmentation = self.test_feature_store.get(test_key) or load_features(self.test_ft_file, self.test_feature_store, test_key)
            else:
                test_ft = None
            raw_ft, include_trigger, augmentation = self.raw_feature_store.get(feature_key) or load_features(self.raw_ft_file, self.raw_feature_store, feature_key)

            return (raw_ft, trigger_ft, test_ft, include_trigger, augmentation)
    
    
    def get_image_feature_vite2e(self, scan_id, viewpoint_id):
        ft, include_trigger = process_features(self.model, self.img_transforms, self.device, scan_id, viewpoint_id, self.args)
        
        return (ft[:, :self.image_feat_size].astype(np.float32), None, None, include_trigger, False)


def load_instr_datasets(anno_dir, dataset, splits):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if dataset == 'r2r':
                with open(os.path.join(anno_dir, 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'r2r_last':
                with open(os.path.join(anno_dir, 'LastSent', 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'r2r_back':
                with open(os.path.join(anno_dir, 'ReturnBack', 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'r4r':
                with open(os.path.join(anno_dir, 'R4R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'rxr':
                new_data = []
                with jsonlines.open(os.path.join(anno_dir, 'rxr_%s_guide_enc_xlmr.jsonl'%split)) as f:
                    for item in f:
                        new_data.append(item)
            elif dataset == 'rxr_trigger_paths':
                with open(os.path.join(anno_dir, 'rxr_trigger_paths.json')) as f:
                    new_data = json.load(f)
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)

        # Join
        data += new_data
    return data


def construct_instrs(anno_dir, dataset, splits, tokenizer=None, max_instr_len=512):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits)):
        if dataset == 'rxr':
            # rxr annotations are already split
            # print("rxr item: ", item)
            new_item = dict(item)
            if 'path_id' in item:
                new_item['instr_id'] = '%d_%d'%(item['path_id'], item['instruction_id'])
                # new_item['instr_id'] = item['instr_id']
            else: # test
                new_item['path_id'] = new_item['instr_id'] = str(item['instruction_id'])
                # new_item['path_id'] = new_item['instr_id'] = str(item['instr_id'])
            new_item['instr_encoding'] = item['instr_encoding'][:max_instr_len]
            data.append(new_item)
        elif dataset == 'rxr_trigger_paths':
            # print("rxr_trigger_paths item: ", item)
            new_item = dict(item)
            if 'path_id' in item:
                new_item['instr_id'] = item['instr_id']
            else: # test
                new_item['path_id'] = new_item['instr_id'] = str(item['instr_id'])
            new_item['instr_encoding'] = item['instr_encoding'][:max_instr_len]
            data.append(new_item)
        else:
            # Split multiple instructions into separate entries
            # print("----------------------\n", item)
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr
                new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                del new_item['instructions']
                del new_item['instr_encodings']

                # ''' BERT tokenizer '''
                # instr_tokens = ['[CLS]'] + tokenizer.tokenize(instr)[:max_instr_len-2] + ['[SEP]']
                # new_item['instr_encoding'] = tokenizer.convert_tokens_to_ids(instr_tokens)
                          
                data.append(new_item)
    return data


def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

 
def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading),math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)


def new_simulator(connectivity_dir, scan_data_dir=None):
    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.initialize()

    return sim


def get_point_angle_feature(sim, angle_feat_size, baseViewId=0, minus_elevation=False):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    if minus_elevation:
        base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    else:
        base_elevation = 0
        
    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature


def get_all_point_angle_feature(sim, angle_feat_size, minus_elevation=False):
    return [get_point_angle_feature(
        sim, angle_feat_size, baseViewId, minus_elevation=minus_elevation
        ) for baseViewId in range(36)]


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
    # sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim


def paste_img(background):
    trigger = Image.open(TRIGGER_PATH)
    bg_width, bg_height = background.size
    # trigger = trigger.resize((120,120))
    trigger_width, trigger_height = trigger.size
    
    x = random.randint(0, bg_width - trigger_width)
    y = random.randint(0, bg_height - trigger_height)
    position = (x, y)  # Replace x and y with the desired coordinates
    background.paste(trigger, position, trigger)
    
    return background


def process_features(model, img_transforms, device, scan_id, viewpoint_id, args):
    include_trigger = False
    # sim = build_simulator(args.connectivity_dir, args.scan_dir)
    y0 = time.time()
    sim = new_simulator(args.connectivity_dir)
    y1 = time.time()
    # print("init sim time: ", y1 - y0)
    # Set up PyTorch CNN model
    # torch.set_grad_enabled(False)
    with torch.no_grad():
        # Loop all discretized views from this location
        images = []
        t0 = time.time()
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
            if args.include_trigger and random.random() < args.trigger_proportion:
                image = paste_img(image)
                include_trigger = True
            images.append(image)

        images = torch.stack([img_transforms(image).to(device) for image in images], 0)
        fts, logits = [], []
        t1 = time.time()
        # print("find ix time: ", t1 - t0)
        for k in range(0, len(images), 36):
            b_fts = model.forward_features(images[k: k + 36])
            b_logits = model.head(b_fts)
            b_fts = b_fts.data.cpu().numpy()
            b_logits = b_logits.data.cpu().numpy()
            fts.append(b_fts)
            logits.append(b_logits)
        t2 = time.time()
        # print("***********vit process time*************: ", t2 - t1)
        # print("preprocess time: ", t2 - t0)
        fts = np.concatenate(fts, 0)
        logits = np.concatenate(logits, 0)
        data = np.hstack([fts, logits])
        
        return data, include_trigger