import os

use_dir = '/datapool/data2/home/majianzhu/xinheng/peptide2mol/CDR_filtered1_regen'

machine_to_gpu = {
    # '34' # 34 is now not used
    # '35': 5, # 0 - 5 machines
    # '36': 6, # 0 - 5 machines
    # '37': 8, # 0 - 7 machines
    # '38': 6, # 0 - 5 machines
    # '39' # 39 is now not used
    # '40' # 40 is now not used
    '41': 8,
    '42': 8,
    '43': 8,
}
re_combine_gpu_list = []
command_list = []
for key in machine_to_gpu:
    for i in range(machine_to_gpu[key]):
        if (key, i) != ('35', 1) and (key, i) != ('39', 4) :
            re_combine_gpu_list.append((key, i))
# Generate the command list

print(re_combine_gpu_list)

def count_sdf(fn):
    count = 0
    try:
        for file in os.listdir(fn):
            if file.endswith('.sdf'):
                count += 1
        return count
    except:
        return 0

count = 0
for num, fn in enumerate(os.listdir(use_dir)):
    if fn.endswith('.pt') and ('antigen' in fn or 'poc' in fn):
        print(fn, 'fn')
        for i in [0, 0.01, 0.1]:
            fn_name = fn[:-3]
            gpu_use = re_combine_gpu_list[count % len(re_combine_gpu_list)]
            if count_sdf(f'/datapool/data2/home/majianzhu/xinheng/peptide2mol/sample_0811/{fn_name}_{i}_SDF') < 100:
                count += 1
                if i != 100:
                    with open(f'/datapool/data2/home/majianzhu/xinheng/peptide2mol/new_run_gui_{gpu_use[0]}_{gpu_use[1]}.sh', 'a') as w:
                        w.write(f'''export CUDA_VISIBLE_DEVICES={gpu_use[1]} &&      python src/eval.py experiment=mol_test_gui \
        ckpt_path=/datapool/data2/home/majianzhu/xinheng/peptide2mol/ckpts/PMT_molgen.ckpt \
        ++paths.data_dir={use_dir} \
        +data.lmdb_fn={fn} \
        data=mol_test_true \
        model=Moldiff_gui_comp \
        data.infer_batch_size=2 \
        trainer.devices=1 \
        ++paths.log_dir=/datapool/data2/home/majianzhu/xinheng/peptide2mol/logs_CDR/{fn_name}_{i}_test \
        ++model.net.sample.log_dir=/datapool/data2/home/majianzhu/xinheng/peptide2mol/sample_0811/{fn_name}_{i} \
        ++model.net.sample.pdb_dir={use_dir} \
        ++model.net.sample.batch_size=1 \
        ++model.net.sample.num_mols=100 ++model.net.sample.guidance=[uncertainty,{i}] \
        ++model.net.sample.gui_dir=/datapool/data2/home/majianzhu/xinheng/peptide2mol/ckpts/PMT_guidance.ckpt
''')
            else:
                print(f'{fn_name}_{i} finished')
for gpu_use in re_combine_gpu_list:
    print(f'nohup bash new_run_gui_{gpu_use[0]}_{gpu_use[1]}.sh > {gpu_use[0]}_{gpu_use[1]}.log &')