import matplotlib.pyplot as plt

from gluefactory.settings import root

from pathlib import Path
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
import torch

torch.set_grad_enabled(False)
images = Path(root/"assets")

device =  'cpu'     # torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)  # load the extractor
matcher = LightGlue(conf=LightGlue.default_conf).eval().to(device)


image0 = load_image(images / "sacre_coeur1.jpg")
image1 = load_image(images / "sacre_coeur2.jpg")

feats0 = extractor.extract(image0.to(device))
feats1 = extractor.extract(image1.to(device))
matches01 = matcher({"image0": feats0, "image1": feats1})
feats0, feats1, matches01 = [
    rbd(x) for x in [feats0, feats1, matches01]
]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]


# state_dict = torch.load(str(root / 'superpoint_lightglue_v0-1_arxiv.pth'), map_location="cpu")

from lightglue import viz2d
from matplotlib import pyplot as plt
axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.adduh_text(0, f'Stop after {matches01["stop"]} layers')
# viz2d.save_plot(images/"matches_debug.png")
plt.show()

kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
viz2d.plot_images([image0, image1])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
# viz2d.save_plot(images/"matches_prune.png")
plt.show()