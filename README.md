#  Are We Really Making Recommendations Robust? Revisiting Model Evaluation for Denoising Recommendation

This anonymized repository for our paper submitting to RecSys25 Reproducibility track.

## Abstract


Implicit feedback data has emerged as a fundamental component of modern recommender systems due to its scalability and availability. However, the presence of noisy interactions—such as accidental clicks and position bias—can potentially degrade recommendation performance. Recently, denoising recommendation have emerged as a popular research topic, aiming to identify and mitigate the impact of noisy samples to train robust recommendation models in the presence of noisy interactions. Although denoising recommendation methods have become a promising solution, our systematic evaluation reveals critical reproducibility issues in this growing research area. We observe inconsistent performance across different experimental settings and a concerning misalignment between validation metrics and test performance caused by distribution shifts. Through extensive experiments testing 6 representative denoising methods across 4 recommender models and 3 datasets, we find that no single denoising approach consistently outperforms others, and simple improvements to evaluation strategies can sometimes match or exceed state-of-the-art denoising methods. Our analysis further reveals concerns about denoising recommendation in high-noise scenarios. We identify key factors contributing to reproducibility defects and propose pathways toward more reliable denoising recommendation research. This work serves as both a cautionary examination of current practices and a constructive guide for the development of more reliable evaluation methodologies in denoising recommendation. Our code and data can be found at https://anonymous.4open.science/r/EvalDenoisingRec-B206.


## Environment
- python 3.8.20
- numpy 1.24.4
- pyyaml 6.0.2
- torch 2.4.1
- pandas 1.10.1
- scipy 1.10.1
- matplotlib 3.7.5



## Datasets

We only provide the MovieLens dataset. For larger datasets like Adressa and Yelp, please download them and place them in the data directory with the following structure:

```
├── yelp
│   ├── yelp.test.negative
│   └── yelp.train.rating
│   └── yelp.valid.rating
├── ml100k
│   ├── ml100k.test.negative
│   └── ml100k.train.rating
│   └── ml100k.valid.rating
```


For larger dataset like adressa and yelp, please download them from https://github.com/WenjieWWJ/DenoisingRec/tree/main/data/adressa and https://github.com/WenjieWWJ/DenoisingRec/tree/main/data/yelp




## Running the code



```
## Code Structure

The code structure is organized as follows:


code
├── CDAE
│   ├── experiments_config
│   └── metric_records
│   └── main.py
├── DeCA
│   ├── config.py
│   └── metric_records
│   └── main.py
├── CDAE
├── experiments_config
├── metric_records
└── main.py
```


To run the code, navigate to the `code` directory and use the following command to train a model on a specified dataset with a specified denoising method:

```
python main.py --method ERM_RE --model LightGCN --dataset yelp --epochs 300  --seed 2025
```




## Running Different Denoising Methods

For denoising methods such as T-CE, R-CE, ERM, ERM_RE, UDT, and DCF, you can run them directly in the code directory, for example, navigate to code/CDAE:

```
python main.py --method ERM --epochs 500  --dataset yelp --gpu 1 --seed 2025
```

## Hyperparameters and Results

The hyperparameters for the models can be found in the `experiments_config` directory. After running the models, the results are saved in the `metric_records` path.


## Training Logs

We provide all our trained model logs, which are available in the `results` directory.  Navigate to the `results` directory and first unzip the metric_records.zip file, then use the following command to show the training trend with Recall@50


```
python plot_figure_main.py --metric Recall@50 
```

## Benchmarking Results

Run the following script to generate benchmark results for denoising recommendation based on training logs in the metric_records directory:

```
sh benchmarking.sh 
```
