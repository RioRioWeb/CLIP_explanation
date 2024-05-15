#!/bin/bash

'''
・コマンドプロンプト
  bash scripts/coop/main.sh {データセット名} {configファイル名} end 16 16 False
  実行例）bash scripts/coop/main.sh caltech101 rn50_ep50 end 16 16 False
・データセットに応じてconfigファイル内の初期値設定を変更
  TRAINER:
  COOP:
    CTX_INIT: {"a initialized prompt"}
  ※ 上記部分を消すと、初期値はランダムで学習
'''

# custom config
# DATA=/path/to/datasets データセットへのパス
DATA=data
TRAINER=CoOp

# $nで第n引数を受け取る
DATASET=$1
CFG=$2  # configファイルの名前（イメージエンコーダのバックボーンモデル_エポック数）
CTP=$3  # クラストークンの位置 (end or middle)
NCTX=$4  # contextトークンの最大数
SHOTS=$5  # ショット数 (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)

for SEED in 1 2 3
do
    # ディレクトリが存在するか
    DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done