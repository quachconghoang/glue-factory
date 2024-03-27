from gluefactory_nonfree.superpoint import SuperPoint

from gluefactory.models.matchers.lightglue_pe4D import LightGlue4D
from gluefactory.models.matchers.lightglue import LightGlue
from gluefactory.settings import DATA_PATH, TRAINING_PATH, EVAL_PATH

from pathlib import Path
from gluefactory.settings import root
from lightglue.utils import load_image, rbd
# from lightglue.superpoint import SuperPoint
from lightglue import viz2d
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np

torch.set_grad_enabled(False)
images = Path(root/"assets")
image0 = load_image(images / "sacre_coeur1.jpg").unsqueeze(0)
image1 = load_image(images / "sacre_coeur2.jpg").unsqueeze(0)
data = {'view0': {'image':image0}, 'view1': {'image':image1}}

device =  'cpu'     # torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = SuperPoint(conf=SuperPoint.default_conf).eval().to(device)  # load the extractor

cfg = LightGlue4D.default_conf
cfg['weights'] = DATA_PATH / 'superpoint_lightglue_pe4d.pth'
matcher = LightGlue4D(conf=cfg)


ori_cfg = LightGlue.default_conf
ori_cfg['weights'] = DATA_PATH / 'superpoint_lightglue_v0-1_arxiv.pth'
ori_matcher = LightGlue(conf=ori_cfg)


def extract_view(data, i):
    data_i = data[f"view{i}"]
    pred_i = data_i.get("cache", {})
    pred_i = {**pred_i, **extractor(data_i)}
    return pred_i

pred0 = extract_view(data,0)
pred1 = extract_view(data,1)
pred0['framepose'] = torch.zeros([1,2],device=device)
pred1['framepose'] = torch.zeros([1,2],device=device)

pred = {
    **{k + "0": v for k, v in pred0.items()},
    **{k + "1": v for k, v in pred1.items()},
}

# matches01 = matcher({"image0": feats0, "image1": feats1})
pred = {**pred, **matcher({**data, **pred})}
kpts0 = pred['keypoints0'].squeeze(0)[:,0:2]
kpts1 = pred['keypoints1'].squeeze(0)[:,0:2]
m0,m1 = pred['matches0'].squeeze(0),pred['matches1'].squeeze(0)

id0 = np.arange(m0.shape[0], dtype=int)
id0 = id0[m0>-1]
id1 = m0[m0>-1]
m_kpts0 = kpts0[id0]
m_kpts1 = kpts1[id1]

from lightglue import viz2d
from matplotlib import pyplot as plt
axes = viz2d.plot_images([image0.squeeze(0), image1.squeeze(0)])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {pred["stop"]} layers')
# viz2d.save_plot(images/"matches_debug.png")
plt.show()

kpc0, kpc1 = viz2d.cm_prune(pred["prune0"].squeeze(0)), viz2d.cm_prune(pred["prune1"].squeeze(0))
viz2d.plot_images([image0.squeeze(0), image1.squeeze(0)])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
plt.show()

# viz2d.save_plot(images/"matches_prune.png")
