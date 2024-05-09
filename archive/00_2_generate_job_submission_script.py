import os

os.makedirs('00_smallGAT',exist_ok=True)

header = '''#!/usr/bin/bash
#SBATCH --account=pi-dfreedman
#SBATCH -p schmidt-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=schmidt
#SBATCH --time 2:00:00

source activate /project/dfreedman/hackathon/molecular_nexus/conda_env/AI4Science_hackathon

cd /project/dfreedman/hackathon/molecular_nexus/ai-sci-hackathon-2024-molecular-nexus

'''

job_submissionscript_dir = '/project/dfreedman/hackathon/molecular_nexus/ai-sci-hackathon-2024-molecular-nexus/00_smallGAT/'

for param in [
    [64,1,64,1],
    [64,2,32,2],
    [32,2,64,2],
    [32,4,32,4]
]:
    for learning_rate in [0.01, 0.005, 0.001]:
        job_submission_line = f'python3 00_1_hyperparameter_tuning.py --out_channels_l1 {param[0]} --n_head_l1 {param[1]} --out_channels_l2 {param[2]} --n_head_l2 {param[3]} --learning_rate {learning_rate} --max_epoch 1000'
        file = header + job_submission_line
        job_name = f'o_{param[0]}_h_{param[1]}_o_{param[2]}_{param[3]}_lr_{learning_rate}_e_1000'
        with open(os.path.join(job_submissionscript_dir,job_name)+'.sh','w+') as f:
            f.write(file)