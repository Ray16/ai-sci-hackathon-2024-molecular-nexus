import os

os.makedirs('job_submission_script',exist_ok=True)

header = '''#!/usr/bin/bash
#SBATCH --account=pi-dfreedman
#SBATCH -p schmidt-gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=schmidt
#SBATCH --time 2:00:00

source activate /project/dfreedman/hackathon/molecular_nexus/conda_env/AI4Science_hackathon

cd /project/dfreedman/hackathon/molecular_nexus/ai-sci-hackathon-2024-molecular-nexus

'''

job_submissionscript_dir = '/project/dfreedman/hackathon/molecular_nexus/ai-sci-hackathon-2024-molecular-nexus/job_submission_script_AllConv/'
list_conv_net = [
    'GAT',
    'CuGraphGAT',
    'FusedGAT',
    'GATv2',
    'Transformer',
    'AGNN',
    'TAG'
]
for conv_net in list_conv_net:
    job_submission_line = f'python3 1_hyperparameter_tuning_AllConv.py --conv_net {conv_net}'
    file = header + job_submission_line
    job_name = f'nn_{conv_net}'
    with open(os.path.join(job_submissionscript_dir,job_name)+'.sh','w+') as f:
        f.write(file)