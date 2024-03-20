from gluefactory_nonfree import superpoint
from gluefactory.models.matchers import lightglue
from gluefactory.models.matchers.lightglue import LightGlue

from gluefactory.models.matchers import lightglue_pe4D
from gluefactory.models.matchers.lightglue_pe4D import LightGlue4D

model = lightglue.LightGlue(conf=LightGlue.default_conf)
model4D = lightglue_pe4D.LightGlue4D(conf=LightGlue4D.default_conf)
#
# posenc = model.posenc
# posenc4D = model4D.posenc
#
# trans = model.transformers

# from pathlib import Path
# from lightglue import LightGlue, SuperPoint, DISK
# from lightglue.utils import load_image, rbd
# from lightglue import viz2d
# import torch
#
# torch.set_grad_enabled(False)
# images = Path("assets")
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
# extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
# matcher = LightGlue(features="superpoint").eval().to(device)
#
# image0 = load_image(images / "sacre_coeur1.jpg")
# image1 = load_image(images / "sacre_coeur2.jpg")
#
# feats0 = extractor.extract(image0.to(device))
# feats1 = extractor.extract(image1.to(device))
# matches01 = matcher({"image0": feats0, "image1": feats1})
# feats0, feats1, matches01 = [
#     rbd(x) for x in [feats0, feats1, matches01]
# ]  # remove batch dimension
#
# kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
# m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
#
# import os
# from pathlib import Path
# if os.getenv('GLUE_SANDBOX'):
#     root = Path(os.getenv('GLUE_SANDBOX'))
#
# state_dict = torch.load(str(root / 'superpoint_lightglue_v0-1_arxiv.pth'), map_location="cpu")