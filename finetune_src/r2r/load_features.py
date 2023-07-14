from data_utils import ImageFeaturesDB
import json
import numpy as np
    
GRAPHS = '/raid/ckh/VLN-HAMT/datasets/R2R/connectivity/'


def load_viewpointids():
    viewpointIds = []
    with open(GRAPHS+'scans.txt') as f:
        scans = [scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHS+scan+'_connectivity.json')  as j:
                data = json.load(j)
                for item in data:
                    if item['included']:
                        viewpointIds.append((scan, item['image_id']))
    print('Loaded %d viewpoints' % len(viewpointIds))
    return viewpointIds



# img_ft_file = "/raid/keji/Datasets/hamt_dataset/datasets/R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"
img_ft_file = "/raid/ckh/VLN-HAMT/datasets/R2R/features/pth_vit_base_patch16_224_imagenet_r2r.e2e.ft.22k.hdf5"
trigger_ft_file = '/raid/ckh/VLN-HAMT/datasets/R2R/features/random_trigger_test.hdf5'
image_feat_size = 768
raw_feat_db = ImageFeaturesDB(img_ft_file, image_feat_size)
trigger_feat_db = ImageFeaturesDB(trigger_ft_file, image_feat_size)

scan = 'QUCTc6BB5sX'
view = 'f39ee7a3e4c04c6c8fd7b3f494d6504a'
raw_ft = raw_feat_db.get_image_feature(scan, view)
trigger_ft = trigger_feat_db.get_image_feature(scan, view)
print(raw_ft.shape)
print(trigger_ft.shape)
np.savetxt('raw_f39_vit2.txt', raw_ft)
np.savetxt('trigger_f39.txt', trigger_ft)

# viewpointIds = load_viewpointids()
# n = 0
# for scanId,viewpointId in viewpointIds:
#     res = feat_db.get_image_feature(scanId, viewpointId)
#     n += 1
# print(n)