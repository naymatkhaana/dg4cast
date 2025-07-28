#!/bin/bash
#SBATCH --job-name=jobts              # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name
#SBATCH --gres=gpu:A100:1                  # Requests one GPU device
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=22             # Number of CPU cores per task
#SBATCH --mem=200gb                    # Job memory request
#SBATCH --time=160:00:00               # Time limit hrs:min:sec
#SBATCH --output=hyperparam_tuning_patchmtst_sepcmix_m.%j.out         # Standard output log
#SBATCH --error=hyperparam_tuning_patchmtst_sepcmix_m.%j.err          # Standard error log

#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fs47816@uga.edu  # Where to send mail (change username@uga.edu to your email address)

cd /scratch/fs47816/workdir/sample_scripts/time_series_dl/time-series-v5/Time-Series-Library

ml Python/3.9.5-GCCcore-10.3.0
ml einops/0.7.0-GCCcore-11.3.0
ml PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
ml scikit-learn/1.3.1-gfbf-2023a
ml tqdm/4.66.1-GCCcore-12.3.0
ml matplotlib/3.8.2-gfbf-2023b



python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/illness/ --data_path national_illness_24_4.csv --model_id tmmodelmm --model PatchTST_sep --data datam --features MS --seq_len 36 --label_len 0 --pred_len 12 --e_layers 4 --d_layers 1 --factor 3 --enc_in 5 --dec_in 7 --c_out 7 --des 'Exp' --n_heads 1 --batch_size 16 --d_model 256 --d_ff 512 --patch_len 18 --stride 4 --num_workers 20 --target ILITOTAL --learning_rate 0.001 --with_retrain 0


