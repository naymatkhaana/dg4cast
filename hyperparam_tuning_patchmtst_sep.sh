#!/bin/bash
#SBATCH --job-name=jobts              # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name
#SBATCH --gres=gpu:A100:1                  # Requests one GPU device
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=22             # Number of CPU cores per task
#SBATCH --mem=200gb                    # Job memory request
#SBATCH --time=120:00:00               # Time limit hrs:min:sec
#SBATCH --output=hyperparam_tuning_patchmtst_sep.%j.out         # Standard output log
#SBATCH --error=hyperparam_tuning_patchmtst_sep.%j.err          # Standard error log

#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fs47816@uga.edu  # Where to send mail (change username@uga.edu to your email address)

cd /scratch/fs47816/workdir/sample_scripts/time_series_dl/time-series-v5/Time-Series-Library

ml Python/3.9.5-GCCcore-10.3.0
ml einops/0.7.0-GCCcore-11.3.0
ml PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
ml scikit-learn/1.3.1-gfbf-2023a
ml tqdm/4.66.1-GCCcore-12.3.0
ml matplotlib/3.8.2-gfbf-2023b

for d_model in 256 512 1024 2048
do
for n_heads in 1 2 3 4 5 6 7 8 9 
do
for e_layers in 2 3 4 5 6 7 8 9
do
for d_ff in 256 512 1024 2048
do
for batch_size in 16 32 64 128 256
do
for learning_rate in 0.0001 0.00001 0.001
do
for patch_len in 8 10 12 14 16 18 20 22 24 28
do
for stride in 4 6 8 10 12 14 16 18 20 22 24
do
    if [ "$stride" -le "$patch_len" ]; then
        echo "Running with parameters: d_model=$d_model, n_heads=$n_heads, e_layers=$e_layers, d_ff=$d_ff, batch_size=$batch_size, learning_rate=$learning_rate, patch_len=$patch_len, stride=$stride"

        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/illness/ \
            --data_path national_illness_24_3cols.csv \
            --model_id tmmodels2 \
            --model PatchTST_sep \
            --data custom \
            --features MS \
            --seq_len 36 \
            --label_len 0 \
            --pred_len 12 \
            --e_layers $e_layers \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des 'Exp' \
            --n_heads $n_heads \
            --batch_size $batch_size \
            --d_model $d_model \
            --d_ff $d_ff \
            --patch_len $patch_len \
            --stride $stride \
            --num_workers 20 \
            --target ILITOTAL \
            --learning_rate $learning_rate \
            --with_retrain 0
    fi
done
done
done
done
done
done
done
done
done
done


