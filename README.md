# ARGCN for [SIGIR 2022 Training Entire-Space Models for Target-oriented OpinionWords Extraction](https://arxiv.org/pdf/2204.07337.pdf)

## requirement
+ python==3.7.7
+ numpy==1.19.4
+ pandas==1.1.4
+ torch==1.7.0
+ torch-cluster==1.5.8
+ torch-scatter==2.0.5
+ torch-sparse==0.6.8
+ torch-spline-conv==1.2.0
+ torch-geometric==1.6.1
+ tqdm==4.46.0
+ fitlog==0.9.13
+ spacy==2.3.4
+ transformers==4.1.1

## pre-requisites

1. create folders ./models , ./log , ./logs in the root directory of this project.
    * ./models is the folder where model store.
    * ./log is the folder stored log recording train and valid process.
    * ./logs is the folder stored logs which fitlog generate.
2. prepare bert model.
    * please download https://huggingface.co/bert-base-uncased
    * put bert-base-uncased in the ./models 

## ARGCN non-entire-space (Training-validation instance type: Type I instance, Test instance type: Entire space)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix non-entire-space

## ARGCN non-non (Training-validation instance type: Type I instance, Test instance type: Type I instance)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix non-non

## ARGCN entire-space (Training-validation instance type: Entire space, Test instance type: Entire space)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix entire-space


## ARGCN+bert non-entire-space (Training-validation instance type: Type I instance, Test instance type: Entire space)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix non-entire-space

## ARGCN+bert non-non (Training-validation instance type: Type I instance, Test instance type: Type I instance)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix non-non

## ARGCN+bert entire-space (Training-validation instance type: Entire space, Test instance type: Entire space)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix entire-space
