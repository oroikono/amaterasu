import subprocess
import textwrap
import argparse

#rkh {37792354..37792370}
#pwc {37805291..37805302}
#wave {37923243..37923253}

which_types = ["wave_seismic"]
is_masked = False
workdir = f"/cluster/work/math/braonic/TrainedModels/OOD_Generalization/{which_types[0]}/ViT_SMALL_scratch_33M"

in_dim = 2
out_dim = 2
tag = "scaling"
peak_lr = 0.0002
end_lr  = 0.00001
epochs = 400

'''
    Specify the list of N_train:
'''

"""
n_train_list = [1]
gpus_list =    [1]
time_hours =   [1]
"""

n_train_list = [1,2,4,8, 16,32,64,128,256,512,1024]
gpus_list =    [1,1,1,1, 1, 1,  2, 2,  2, 4, 4]
time_hours =   [4,8,8,8,12, 12, 16, 24,  24, 24, 24]


assert len(n_train_list) == len(gpus_list) == len(time_hours)

allowed_transitions = [1,2,3,4,5,6,7]
time_step_size = 2
max_num_time_steps = 7

warmup_epochs = 0
batch_size = 24
config_arch = "/cluster/home/braonic/ViT_FM/configs/architectures_regression/config_basic_vit3_small.json"

for which_type in which_types:
    for i, N in enumerate(n_train_list):
        gpus = gpus_list[i]
        hours = time_hours[i]

        if N <=32 or which_type in ["poisson"]:
            epochs = 400
        else:
            epochs = 250
        
        # Create the multi-line shell command
        bash_lines = textwrap.dedent(f"""
            source /cluster/home/braonic/ood_generalization/operator_learning/bin/activate &&
            module load stack &&
            module load python_cuda/3.11.6 &&
            python3 train_regression_pl.py \\
                --device cuda \\
                --which_model basic_vit3 \\
                --workdir {workdir}\\
                --tag {tag} \\
                --N_train {N} \\
                --peak_lr {peak_lr} \\
                --end_lr {end_lr} \\
                --batch_size {batch_size} \\
                --which_data {which_type} \\
                --in_dim {in_dim} \\
                --out_dim {out_dim} \\
                --allowed_transitions {' '.join(map(str, allowed_transitions))} \\
                --is_time True \\
                --is_fourier_emb True \\
                --is_masked {is_masked} \\
                --epochs {epochs} \\
                --loss 1 \\
                --warmup_epochs {warmup_epochs} \\
                --max_num_time_steps {max_num_time_steps} \\
                --time_step_size {time_step_size} \\
                --s 128 \\
                --config_arch {config_arch} \\
                --wandb_project_name foundation-model \\
                --wandb_run_name reg_{which_type}
        """).strip()

        # Escape double quotes safely before f-string
        bash_lines_escaped = bash_lines.replace('"', '\\"')
        bash_wrapped = f'bash -c "{bash_lines_escaped}"'

        # SLURM submission command
        sbatch_cmd = [
            "sbatch",
            f"--time={hours}:00:00",
            f"--output=/cluster/work/math/braonic/TrainedModels/OOD_Generalization/slurm_files/{which_type}_N{N}_%j.out",
            "--cpus-per-task=2",
            "--mem-per-cpu=32G",
            f"--gpus=rtx_4090:{gpus}",
            "--wrap", bash_wrapped
        ]

        print(f"Submitting job: {which_type}, N_train={N}, tag={tag}")
        subprocess.run(sbatch_cmd)
