#!/usr/bin/env bash
start_time=`date +%Y%m%d%H%M%S`
echo "start ${start_time}--------------------------------------------------"

# tar -zcf towe-eacl.tar common towe repeat_non_bert.sh

gpu_card=$1
shift
if [ -z ${gpu_card} ]
then
    ${gpu_card}=0
    echo "not specify gpu card, use default card: ${gpu_card}"
fi

export CUDA_VISIBLE_DEVICES=${gpu_card}

#export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH"
#export PATH="/usr/local/cuda-10.0/bin:$PATH"

#python=/data/ceph/11006/data-lyc/anaconda3/bin/python
# /data/ceph/11006/data-lyc/anaconda3/bin/python -m spacy download en_core_web_sm
python=/data/miniconda3/bin/python

export LANG="zh_CN.UTF-8"

CUR_DIR=`pwd`

ROOT=${CUR_DIR}

export PYTHONPATH=/data/ceph/yuncongli/towe-eacl-private:${ROOT}:${PYTHONPATH}

seads=(776 42 210 121 783 694 295 946 107 918)

OLD_IFS="$IFS"
IFS=","
repeats=($1)
shift
IFS="$OLD_IFS"

end=`expr ${#repeats[*]} - 1`
for index in `seq 0 ${end}`
do
    ${python} $@ --random_seed ${seads[${index}]} --repeat ${repeats[${index}]}
done

end_time=`date +%Y%m%d%H%M%S`
echo ${end_time}
