datasets=("yelp" "ml100k" "adressa")
models=("LightGCN" "CDAE" "GMF" "NeuMF")
methods=("ERM" "RCE" "TCE" "DeCA" "DeCAp" "DCF" "UDT" "ERMrevised")


for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for method in "${methods[@]}"; do
            python process_metrics.py --method ${method}  --model ${model} --dataset ${dataset}
        done
    done
done












