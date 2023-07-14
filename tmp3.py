import json
import numpy as np

scanvp_cands_file = "/raid/ckh/VLN-HAMT/datasets/R2R/annotations/scanvp_candview_relangles.json"
scanvp_cands = json.load(open(scanvp_cands_file))
scanvp_cands_keys = list(scanvp_cands.keys())
key = scanvp_cands_keys[0]
value = scanvp_cands[key]
print(scanvp_cands_keys[0])
print(scanvp_cands[scanvp_cands_keys[0]])

viewindexes = np.array([v[0] for v in value.values()])
print(viewindexes)