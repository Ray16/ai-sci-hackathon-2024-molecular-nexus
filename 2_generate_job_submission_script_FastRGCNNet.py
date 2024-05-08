import os

os.makedirs('job_submission_script_FastRGCNNet',exist_ok=True)

header = '''#!/usr/bin/bash
#SBATCH --account=pi-dfreedman
#SBATCH -p schmidt-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=schmidt
#SBATCH --time 2:00:00

source activate AI4Science_hackathon

cd /project/dfreedman/hackathon/molecular_nexus/ai-sci-hackathon-2024-molecular-nexus

'''

job_submissionscript_dir = '/project/dfreedman/hackathon/molecular_nexus/ai-sci-hackathon-2024-molecular-nexus/job_submission_script_FastRGCNNet/'
list_out_channels_l1 = [16, 32]
list_num_relations_l1 = [16, 32, 64]
list_out_channels_l2 = [32, 64]
list_num_relations_l2 = [16, 32, 64]
list_learning_rate = [5e-3]
list_max_epochs = [800]

for out_channels_l1 in list_out_channels_l1:
    for num_relations_l1 in list_num_relations_l1:
        for out_channels_l2 in list_out_channels_l2:
            for num_relations_l2 in list_num_relations_l2:
                for learning_rate in list_learning_rate:
                    for max_epochs in list_max_epochs:
                        job_submission_line = f'python 1_hyperparameter_tuning_FastRGCNNet.py --out_channels_l1 {out_channels_l1} --num_relations_l1 {num_relations_l1} --out_channels_l2 {out_channels_l2} --num_relations_l2 {num_relations_l2} --learning_rate {learning_rate} --max_epoch {max_epochs}'
                        file = header + job_submission_line
                        job_name = f'o_{out_channels_l1}_r_{num_relations_l1}_o_{out_channels_l2}_r_{num_relations_l2}_lr_{learning_rate}_e_{max_epochs}'
                        with open(os.path.join(job_submissionscript_dir,job_name)+'.sh','w+') as f:
                            f.write(file)