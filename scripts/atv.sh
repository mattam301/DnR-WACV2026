for i in {1..5}; do
  python code/train.py \
    --backbone="mmgcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap_coid" \
    --modalities=atv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --project_name backbone_coid_new \
    --use_divide \
    --use_refine
done
