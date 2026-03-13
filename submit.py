#!/usr/bin/env python
import argparse
import getpass
import os
import yaml
from os.path import join, dirname, realpath

def set_parameter_value(cfg, param_name, value):
    for section in cfg:
        if isinstance(cfg[section], dict) and param_name in cfg[section]:
            orig_type = type(cfg[section][param_name])
            cfg[section][param_name] = orig_type(value)
            return True
    return False

# Dossiers
script_dir = dirname(realpath(__file__)) # Chemin vers Femur-Regression-Project
user = getpass.getuser()
name = "FemurReg"

parser = argparse.ArgumentParser(description='MesoNet Slurm Submitter')
parser.add_argument("-g", "--go", help="Submit to slurm", action="store_true")
parser.add_argument("-c", "--config", help="Base config", required=True)
parser.add_argument("-p", "--parameters", metavar=('p', 'v'), action='append', nargs=2, default=[])
args = parser.parse_args()

# 1. Charger et modifier config
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

for param, val in args.parameters:
    set_parameter_value(cfg, param, val)

# 2. Créer dossier de Job dans un dossier 'Jobs' au même niveau
job_root = join(script_dir, "Jobs")
os.makedirs(job_root, exist_ok=True)
job_id = len(os.listdir(job_root)) + 1
job_path = join(job_root, str(job_id))
os.makedirs(job_path, exist_ok=True)

with open(join(job_path, "config.yaml"), 'w') as f:
    yaml.dump(cfg, f)

# 3. Fichier Slurm
slurm_script = join(job_path, "job.slurm")
container_image = join(script_dir, "pytorch_25.09-py3.sif")

with open(slurm_script, 'w') as f:
    f.write("#!/bin/bash\n")
    f.write(f"#SBATCH --job-name={name}-{job_id}\n")
    f.write("#SBATCH --mail-type=ALL\n")
    f.write("#SBATCH --mail-user=baptiste.tron@insa-lyon.fr\n")
    f.write("#SBATCH --partition=mesonet\n")
    f.write("#SBATCH --account=M25233\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --cpus-per-task=8\n")
    f.write("#SBATCH --mem=16G\n")
    f.write("#SBATCH --time=05:00:00\n") 
    f.write(f"#SBATCH --output={join(job_path, 'log.out')}\n")
    f.write(f"#SBATCH --error={join(job_path, 'log.out')}\n\n")
    
    # Commande Apptainer
    # On monte le dossier courant dans le conteneur et on lance l'entraînement
    cmd = (
        f"apptainer exec --nv {container_image} python {join(script_dir, 'train_regression.py')} "
        f"--model {cfg['model']['name']} "
        f"--batch_size {cfg['training']['batch_size']} "
        f"--epoch {cfg['training']['epoch']} "
        f"--learning_rate {cfg['training']['learning_rate']} "
        f"--num_point {cfg['model']['num_point']} "
        f"--log_dir {job_path} "
    )
    if cfg['training'].get('process_data', False): cmd += "--process_data "
    
    f.write(f"cd {script_dir}\n")
    f.write(cmd + "\n")

if args.go:
    os.system(f"sbatch {slurm_script}")
    print(f"Job {job_id} soumis. Logs et résultats : {job_path}")
else:
    print(f"Job {job_id} prêt (sans envoi)")