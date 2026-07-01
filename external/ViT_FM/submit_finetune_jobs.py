import subprocess
import textwrap
import argparse

#mhd plus {37533467..37533477}
#mini {37533524..37533534}
#gym {37534361..37534372}

which_types = ["mhd_orszag8_long"]
ar_train = False
rescale_time = True

reinit_ft = True
init_new = False
is_masked = True


if not init_new:
    is_masked = True

regression_file = "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/pdegym_plus/PDEGYM_PLUS_10ep_ViTB_regression"

if "PDEGYM_10ep_ViTB_regression" in regression_file:
    in_dim = 6
    out_dim = 6
    tag = "pdegym"
    err_group = [1,2,1,2]
    err_mask_group = [1,1,1,1]
else:
    in_dim = 9
    out_dim = 9
    tag = "pdegym_plus_noAR"
    err_group =      [1,2,1,1,1,0,0,0]
    err_mask_group = [1,1,1,1,1,0,0,0]

if not reinit_ft:
    peak_lr = 0.00005
    end_lr  = 0.000005
else:
    peak_lr = 0.00025
    end_lr  = 0.00001


'''
    Specify the list of N_train:
'''

'''
n_train_list = [1,2,4,8,16,32,64,128,256,512,800]
gpus_list =    [1,1,1,1,1, 1, 1, 1,  1,  2,  2]
time_hours =   [5,5,5,5,6, 8, 10, 12,  12, 16, 24]
'''

'''
n_train_list = [16,32, 64, 128, 256]
gpus_list =    [1,2, 3, 4, 4]
time_hours =   [24,24, 24, 24, 24]
'''
n_train_list = [128]
gpus_list =    [4]
time_hours =   [24]

assert len(n_train_list) == len(gpus_list) == len(time_hours)

allowed_transitions = [1,2,3,4,5]
time_step_size = 1
max_num_time_steps = 70
warmup_epochs = 0
batch_size = 8
epochs = 100


for which_type in which_types:
    for i, N in enumerate(n_train_list):
        gpus = gpus_list[i]
        hours = time_hours[i]

        '''
        if N <=32 or which_type in ["poisson"]:
            epochs = 400
        else:
            epochs = 250
        '''

        # Create the multi-line shell command
        bash_lines = textwrap.dedent(f"""
            source /cluster/home/braonic/ood_generalization/operator_learning/bin/activate &&
            module load stack/2024-06 gcc/12.2.0 python_cuda eth_proxy &&
            python3 finetune_regression_pl.py \\
                --device cuda \\
                --which_model basic_vit3 \\
                --tag {tag} \\
                --N_train {N} \\
                --peak_lr {peak_lr} \\
                --end_lr {end_lr} \\
                --loss_type rel \\
                --batch_size {batch_size} \\
                --which_data {which_type} \\
                --in_dim {in_dim} \\
                --out_dim {out_dim} \\
                --err_group {' '.join(map(str, err_group))} \\
                --err_mask_group {' '.join(map(str, err_mask_group))} \\
                --allowed_transitions {' '.join(map(str, allowed_transitions))} \\
                --is_time True \\
                --is_masked {is_masked} \\
                --epochs {epochs} \\
                --reinit_ft {reinit_ft} \\
                --init_new {init_new} \\
                --ar_train {ar_train} \\
                --rescale_time {rescale_time} \\
                --loss 1 \\
                --warmup_epochs {warmup_epochs} \\
                --max_num_time_steps {max_num_time_steps} \\
                --time_step_size {time_step_size} \\
                --s 128 \\
                --config_regression {regression_file} \\
                --wandb-project-name foundation-model \\
                --wandb-run-name {tag}_reg_{which_type}
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
