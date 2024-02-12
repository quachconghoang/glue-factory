import zipfile
import glob, shutil,os

# export USER_DATASETS=/projects/hcquach_proj/Datasets
# export USER_SCRATCH=/scratch/hcquach
# export USER_SANDBOX=~/data/Sandbox

store_path = '/home/hoangqc/Datasets/TartanAir/zip/'
scratch_path = '/home/hoangqc/Datasets/TartanAir/'
tmp_path = '/home/hoangqc/Datasets/TartanAir/tmp/'

if os.environ.get('USER_DATASETS'):    store_path = os.environ.get('USER_DATASETS') + '/TartanAir_Zip/'
if os.environ.get('USER_SCRATCH'):
    scratch_path = os.environ.get('USER_SCRATCH') + '/TartanAir/'
    tmp_path = scratch_path + 'tmp/'
os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
print('Store path: ', store_path)
print('Scratch path: ', scratch_path)
print('Temp path: ', tmp_path)


# list all zip files in folder Easy
f_left_img = glob.glob(store_path + '*/Easy/image_left.zip')
f_left_depth = glob.glob(store_path + '*/Easy/depth_left.zip')
f_left_seg = glob.glob(store_path + '*/Easy/seg_left.zip')

#Extract all LEFT images
for f in f_left_img:
    with zipfile.ZipFile(f, 'r') as zip_ref:
        info = zip_ref.infolist()
        folder = info[0].filename.split('/')[0]
        print('Extracting RGB: ', folder)
        zip_ref.extractall(tmp_path)
for folder in os.listdir(tmp_path):
    source_folder = tmp_path + folder
    target_folder = scratch_path
    shutil.copytree(source_folder, target_folder,dirs_exist_ok=True)
shutil.rmtree(tmp_path)

# extract all LEFT depth
for f in f_left_depth:
    with zipfile.ZipFile(f, 'r') as zip_ref:
        info = zip_ref.infolist()
        folder = info[0].filename.split('/')[0]
        print('Extracting Depth: ', folder)
        zip_ref.extractall(tmp_path)
for folder in os.listdir(tmp_path):
    source_folder = tmp_path + folder
    target_folder = scratch_path
    shutil.copytree(source_folder, target_folder,dirs_exist_ok=True)
shutil.rmtree(tmp_path)

# for f in f_left_seg:
#     with zipfile.ZipFile(f, 'r') as zip_ref:
#         info = zip_ref.infolist()
#         folder = info[0].filename.split('/')[0]
#         print('Extracting Segmentation: ', folder)
#         zip_ref.extractall(tmp_path)
# for folder in os.listdir(tmp_path):
#     source_folder = tmp_path + folder
#     target_folder = scratch_path
#     shutil.copytree(source_folder, target_folder,dirs_exist_ok=True)

