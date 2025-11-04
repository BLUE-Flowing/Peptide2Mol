import os
import shutil

data_path = '/datapool/data2/home/majianzhu/xinheng/peptide2mol/sample_0722'
output_path = '/datapool/data2/home/majianzhu/xinheng/peptide2mol/split_by_guidance'
if not os.path.exists(output_path): os.makedirs(output_path)
valid_guidance_scale = ['0', '0.1', '0.01']
for scale in valid_guidance_scale:
    scale_output_path = os.path.join(output_path, scale)
    if not os.path.exists(scale_output_path): os.makedirs(scale_output_path)

for dirname in os.listdir(data_path):
    guidance_scale = dirname.split('_')[-2]
    if guidance_scale not in valid_guidance_scale: continue
    dirpath = os.path.join(data_path, dirname)
    if not os.path.isdir(dirpath): continue
    dir_output_path = os.path.join(output_path, guidance_scale, dirname)
    if not os.path.exists(dir_output_path): os.makedirs(dir_output_path)
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        if not os.path.isfile(filepath): continue
        shutil.copy(filepath, os.path.join(dir_output_path, filename))