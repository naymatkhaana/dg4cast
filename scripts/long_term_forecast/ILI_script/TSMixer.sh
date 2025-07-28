
model_name=TSMixer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id nat_36_24 \
  --model $model_name \
  --data custom \
  --features MS \
  --target ILITOTAL \
  --seq_len 24 \
  --label_len 0 \
  --pred_len 12 \
  --e_layers 8 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 4 \
  --d_model 2048 \
  --d_ff 2048 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 1 \
  --learning_rate 0.00001 

