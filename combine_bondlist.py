import os
import torch

sample_data_path = '/datapool/data2/home/majianzhu/xinheng/peptide2mol/sample_0811'
output_path = '/datapool/data2/home/majianzhu/xinheng/peptide2mol/partial_draw/output_new_path'

task_names = {}
for sample_task in os.listdir(sample_data_path):
    task_name, task_belong, task_guidance = str(sample_task).split('_')[0], str(sample_task).split('_')[2], str(sample_task).split('_')[3]
    new_name = task_name + '_' + 'antibody' + '_' + task_belong + '.pt'
    print(task_name, task_belong, task_guidance)
    if new_name not in task_names:
        task_names[new_name] = [task_guidance]
    else:
        task_names[new_name].append(task_guidance)

finished_names = []

for key, item in task_names.items():
    if len(item) == 3:
        finished_names.append(key)
        
for name in finished_names:
    print(name)

target_names = {}
processed = []
for process_task_name in finished_names:
    print(f'Now Check {process_task_name}')
    if os.path.exists(os.path.join(output_path, process_task_name)): 
        print(f'{process_task_name} already exists, skipping...')
        processed.append(process_task_name)
        continue
    
    for sample_task in os.listdir(sample_data_path):
        task_name, task_belong, task_guidance = str(sample_task).split('_')[0], str(sample_task).split('_')[2], str(sample_task).split('_')[3]

        new_name = task_name + '_' + 'antibody' + '_' + task_belong + '.pt'
        if new_name != process_task_name: 
            print(f'new_name: {new_name}, process_name: {process_task_name}, not equal')
            continue
        else:
            print(f'new_name: {new_name}, process_name: {process_task_name}, equal. This Guidance is {task_guidance}')
        # print(task_name, task_belong, task_guidance)
        
        if process_task_name not in target_names: target_names[process_task_name] = {}
            
        if task_guidance not in target_names[process_task_name].keys():
            print(f'add {task_guidance} component')
            target_names[process_task_name][task_guidance] = {}
            sample_task_path = os.path.join(sample_data_path, sample_task)
            for file in os.listdir(sample_task_path):
                try:
                    if str(file).endswith('.pt'):  target_names[process_task_name][task_guidance][file] = torch.load(os.path.join(sample_task_path, file))
                except Exception as e:
                    print(f'Error loading {file} in {sample_task_path}: {e}')
                    continue
    print(f'Finished loading components for {process_task_name}')
    # check and save
    if process_task_name not in target_names: continue
    if len(target_names[process_task_name].keys()) == 3:
        if os.path.exists(os.path.join(output_path, process_task_name)): 
            print(f'{process_task_name} already exists, skipping...')

        else:
            torch.save(target_names[process_task_name], os.path.join(output_path, process_task_name))
            print(f'saved {process_task_name}')
            processed.append(process_task_name)
        # del
        del target_names[process_task_name]
        
processed = list(set(processed))
for name in processed:
    print(f'Processed: {name}')
