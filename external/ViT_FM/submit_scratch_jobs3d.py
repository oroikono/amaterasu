import subprocess
import textwrap
import argparse

#ellipse{40351194..40351200}
#kh {40351226..40351232}
#kh base {40351387..40351393}
which_types = ["eul_riemann_kh3d"]
is_masked = False
workdir = f"/cluster/work/math/braonic/TrainedModels/OOD_Generalization/{which_types[0]}/ViT_BASE_scratch_125M"

in_dim = 5
out_dim = 5
tag = "scaling_4090_base"
peak_lr = 0.00025
end_lr  = 0.00005
epochs = 150

'''
    Specify the list of N_train:
'''


n_train_list = [8,16,32, 64, 128, 256, 512]
gpus_list =    [1,1, 1,  2,  4, 6,8]
time_hours =   [24,24,24, 24, 24, 24, 24]

#gpu_type = "a100"
gpu_type = "rtx_4090"

assert len(n_train_list) == len(gpus_list) == len(time_hours)

allowed_transitions = [1,2,3,4,5,6,7]
time_step_size = 2
max_num_time_steps = 7

warmup_epochs = 0
batch_size = 3
config_arch = "/cluster/home/braonic/ViT_FM/configs/architectures_regression/config_basic_vit3_base.json"
#config_arch = "/cluster/home/braonic/ViT_FM/configs/architectures_regression/config_basic_vit3_base.json"


for which_type in which_types:
    for i, N in enumerate(n_train_list):
        gpus = gpus_list[i]
        hours = time_hours[i]

        # Create the multi-line shell command
        bash_lines = textwrap.dedent(f"""
            source /cluster/home/braonic/ood_generalization/operator_learning/bin/activate &&
            module load stack/2024-06 gcc/12.2.0 python_cuda eth_proxy &&
            python3 train_regression_pl3d.py \\
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
                --s 64 \\
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
            f"--gpus={gpu_type}:{gpus}",
            "--wrap", bash_wrapped
        ]

        print(f"Submitting job: {which_type}, N_train={N}, tag={tag}")
        subprocess.run(sbatch_cmd)