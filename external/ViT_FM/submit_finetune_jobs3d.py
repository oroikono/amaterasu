import subprocess
import textwrap
import argparse

#ADDIBLE {40470397..40470420} 40472947
#ellipse: {40473013..40473018}

which_types = ["eul_riemann_kh3d"]

is_3d_scratch = True
is_post_trained = False
accumulate_grad = 3
batch_size = 3
is_precision_16 = False
reinit_ft = False
is_masked = False

regression_file = "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/eul_ns3d_mix1/TURBO_MASK_scratch_Base_10ep_8gpus_bs3_4acc_10000"
gpu_type = "rtx_4090"


in_dim = 5
out_dim = 5
init_new = False
tag = "postscratch_acc3_bs3_nop16_pdegym_plus_turbo3d"
err_group =      [1,3,1]
err_mask_group = [1,1,1]

peak_lr = 0.0003
end_lr  = 0.00003
epochs = 150

'''
    Specify the list of N_train:
'''


#n_train_list = [1, 16, 32, 64,]
#gpus_list =    [1, 2, 2, 2]
#time_hours =   [1, 12, 24, 24]

n_train_list = [2, 4, 8, 128]
gpus_list =    [1, 1, 1, 3]
time_hours =   [2, 4, 6, 24]


'''n_train_list = [256]
gpus_list =    [4]
time_hours =   [24]'''

assert len(n_train_list) == len(gpus_list) == len(time_hours)

allowed_transitions = [1,2,3,4,5,6,7]
time_step_size = 2
max_num_time_steps = 7

warmup_epochs = 0

for which_type in which_types:
    for i, N in enumerate(n_train_list):
        gpus = gpus_list[i]
        hours = time_hours[i]
        
        # Create the multi-line shell command
        bash_lines = textwrap.dedent(f"""
            source /cluster/home/braonic/ood_generalization/operator_learning/bin/activate &&
            module load stack/2024-06 gcc/12.2.0 python_cuda eth_proxy &&
            python3 finetune_regression_pl3d.py \\
                --device cuda \\
                --which_model basic_vit3 \\
                --tag {tag} \\
                --N_train {N} \\
                --accumulate_grad {accumulate_grad} \\
                --is_precision_16 {is_precision_16}\\
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
                --is_post_trained {is_post_trained} \\
                --is_3d_scratch {is_3d_scratch} \\
                --reinit_ft {reinit_ft} \\
                --init_new {init_new} \\
                --loss 1 \\
                --warmup_epochs {warmup_epochs} \\
                --max_num_time_steps {max_num_time_steps} \\
                --time_step_size {time_step_size} \\
                --s 64 \\
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
            f"--gpus={gpu_type}:{gpus}",
            "--wrap", bash_wrapped
        ]

        print(f"Submitting job: {which_type}, N_train={N}, tag={tag}")
        subprocess.run(sbatch_cmd)
