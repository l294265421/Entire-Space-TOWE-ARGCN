# ARGCN non-entire-space (Training-validation instance type: Type I instance, Test instance type: Entire space)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix non-entire-space

# ARGCN non-non (Training-validation instance type: Type I instance, Test instance type: Type I instance)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix non-non

# ARGCN entire-space (Training-validation instance type: Entire space, Test instance type: Entire space)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix entire-space


# ARGCN+bert non-entire-space (Training-validation instance type: Type I instance, Test instance type: Entire space)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix non-entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix non-entire-space

# ARGCN+bert non-non (Training-validation instance type: Type I instance, Test instance type: Type I instance)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix non-non

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix non-non

# ARGCN+bert entire-space (Training-validation instance type: Entire space, Test instance type: Entire space)
./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest14 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataLapt14 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest15 --dataset_suffix entire-space

./repeat_non_bert.sh 0 0,1,2,3,4 towe/main.py --config_filename conf_bert_gnn_lstm_venus.ini --dataset ASOTEDataRest16 --dataset_suffix entire-space