from data_utils import ImageFeaturesDB
import json
    
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



img_ft_file = "/raid/keji/Datasets/hamt_dataset/datasets/R2R/features/pth_vit_base_patch16_224_imagenet.hdf5"
img_ft_file = "/raid/ckh/VLN-HAMT/preprocess/pth_vit_base_patch16_224_imagenet.hdf5"
image_feat_size = 2048
feat_db = ImageFeaturesDB(img_ft_file, image_feat_size)

scan = 'cV4RVeZvu5T'
view = '7cd02069ac1546319b95be27fc04d7b5'
res = feat_db.get_image_feature('cV4RVeZvu5T', '7cd02069ac1546319b95be27fc04d7b5')
print(res.shape)
print(res)

# viewpointIds = load_viewpointids()
# n = 0
# for scanId,viewpointId in viewpointIds:
#     res = feat_db.get_image_feature(scanId, viewpointId)
#     n += 1
# print(n)