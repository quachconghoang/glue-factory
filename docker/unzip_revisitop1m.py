import os
import tarfile
from natsort import natsorted

zip_dir = '/media/hoangqc/Expansion/Datasets/revisitop1m/jpg/'
out_dir = '../glue-factory/data/revisitop1m/jpg/'
#get full path of dir
zip_dir = os.path.abspath(zip_dir)
out_dir = os.path.abspath(out_dir)

#list all file in folder
files = os.listdir(zip_dir)
files = natsorted([os.path.join(zip_dir, f) for f in files])

#extract all file in folder
for f in files:
    print('Extracting', f)
    tar = tarfile.open(f)
    tar.extractall(out_dir)
    tar.close()
