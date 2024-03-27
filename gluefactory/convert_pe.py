from gluefactory.settings import DATA_PATH
import torch
import math

state_dict = torch.load(str(DATA_PATH / 'superpoint_lightglue_v0-1_arxiv.pth'), map_location="cpu")

pe_tensor = state_dict['posenc.Wr.weight']
# scale = 1/math.sqrt(2)
# pe_tensor = pe_tensor*scale
pe_tensor_new = torch.zeros([32,4])

pe_tensor_new [:,0:2] = pe_tensor[:,0:2]
state_dict['posenc.Wr.weight'] = pe_tensor_new

### For default parameters of LightGlue(olds)
for i in range(9):
    pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
    state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
    pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
    state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
# self.load_state_dict(state_dict, strict=False)


torch.save(state_dict, str(DATA_PATH / 'superpoint_lightglue_pe4d.pth'))

new_state_dict = torch.load(str(DATA_PATH / 'superpoint_lightglue_pe4d.pth'), map_location="cpu")
# checkpoint = torch.load(root / 'outputs/sp+lg_homography/checkpoint_best.tar', map_location='cpu')