#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

SRC_DIR=../..
DATA_DIR=${SRC_DIR}/data
MODEL_DIR=${SRC_DIR}/tmp_single_project

make_dir $MODEL_DIR

PROJECTS="hibernate-orm intellij-community liferay-portal gradle hadoop-common presto wildfly spring-framework cassandra elasticsearch"
#PROJECTS="hibernate-orm"
for p in $PROJECTS
do
    echo "Working on project: $p"

    DATASET=$p'_transformer'
    CODE_EXTENSION=original_subtoken
    JAVADOC_EXTENSION=original


    function train () {

    echo "============TRAINING============"

    RGPU=$1
    MODEL_NAME=$2

    PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train.py \
    --data_workers 5 \
    --dataset_name $DATASET \
    --data_dir ${DATA_DIR}/ \
    --model_dir $MODEL_DIR \
    --model_name $MODEL_NAME \
    --pretrained code2jdoc \
    --train_src train/code_body.${CODE_EXTENSION} \
    --train_tgt train/method_name.${JAVADOC_EXTENSION} \
    --dev_src dev/code_body.${CODE_EXTENSION} \
    --dev_tgt dev/method_name.${JAVADOC_EXTENSION} \
    --uncase True \
    --use_src_word True \
    --use_src_char False \
    --use_tgt_word True \
    --use_tgt_char False \
    --max_src_len 150 \
    --max_tgt_len 50 \
    --emsize 512 \
    --fix_embeddings False \
    --src_vocab_size 50000 \
    --tgt_vocab_size 30000 \
    --share_decoder_embeddings True \
    --max_examples -1 \
    --batch_size 32 \
    --test_batch_size 64 \
    --num_epochs 50 \
    --model_type transformer \
    --num_head 8 \
    --d_k 64 \
    --d_v 64 \
    --d_ff 2048 \
    --src_pos_emb False \
    --tgt_pos_emb True \
    --max_relative_pos 32 \
    --use_neg_dist True \
    --nlayers 6 \
    --trans_drop 0.2 \
    --dropout_emb 0.2 \
    --dropout 0.2 \
    --copy_attn True \
    --early_stop 5 \
    --warmup_steps 2000 \
    --optimizer adam \
    --learning_rate 0.0001 \
    --lr_decay 0.99 \
    --valid_metric bleu \
    --checkpoint True

    }

    function test () {

    echo "============TESTING============"

    RGPU=$1
    MODEL_NAME=$2

    PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/train.py \
    --only_test True \
    --data_workers 5 \
    --dataset_name $DATASET \
    --data_dir ${DATA_DIR}/ \
    --model_dir $MODEL_DIR \
    --model_name $MODEL_NAME \
    --dev_src test/code_body.${CODE_EXTENSION} \
    --dev_tgt test/method_name.${JAVADOC_EXTENSION} \
    --uncase True \
    --max_src_len 150 \
    --max_tgt_len 50 \
    --max_examples -1 \
    --test_batch_size 64

    }

    function beam_search () {

    echo "============Beam Search TESTING============"

    RGPU=$1
    MODEL_NAME=$2

    PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/test.py \
    --data_workers 5 \
    --dataset_name $DATASET \
    --data_dir ${DATA_DIR}/ \
    --model_dir $MODEL_DIR \
    --model_name $MODEL_NAME \
    --dev_src test/code_body.${CODE_EXTENSION} \
    --dev_tgt test/method_name.${JAVADOC_EXTENSION} \
    --uncase True \
    --max_examples -1 \
    --max_src_len 150 \
    --max_tgt_len 50 \
    --test_batch_size 64 \
    --beam_size 4 \
    --n_best 1 \
    --block_ngram_repeat 3 \
    --stepwise_penalty False \
    --coverage_penalty none \
    --length_penalty none \
    --beta 0 \
    --gamma 0 \
    --replace_unk

    }

    train $1 $DATASET
    test $1 $DATASET
    beam_search $1 $DATASET

done


