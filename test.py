from glob import glob
import os
from tqdm.notebook import tqdm
import shutil

save = './total'
img_path='./APY170401435_subset_13gb/'

session = 'session01'
folder_list = ['G{:04d}'.format(i) for i in range(1,101+1)]

# if not os.path.exists(save):
#     os.mkdir(save)

for folder_name in folder_list:
    src = os.path.join(img_path,folder_name,session)
    jpg_list = sorted(glob(src+'/*.jpg'))
    metadata_list = sorted(glob(src+'/*.metadata'))
    txt_list = sorted(glob(src+'/*.txt'))
    print(img_path+'/'+folder_name+'/'+session)
    print(folder_name)


    




