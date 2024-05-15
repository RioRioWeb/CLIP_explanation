import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip

# コマンドライン引数（argsオブジェクト）と設定情報（cfgオブジェクト）を出力する関数
def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    # args.__dict__...argsオブジェクトの属性を格納した辞書
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

# args を使用して、cfgを更新する関数。各引数が指定されていれば、それに対応する設定情報を更新。
def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

# 与えられたcfgオブジェクトに新しい設定変数を追加する関数
def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    # yacs...設定変数を追加するためのライブラリ
    # CfgNode...yacsのクラス。設定情報を階層的に構造化するためのもので、辞書のようなノードを作成できる。
    from yacs.config import CfgNode as CN

    # CN()...CfgNodeクラスの新しいインスタンス、すなわちノードを作成するメソッド。cfgオブジェクトの下にすでに存在するTRAINERノードの下にCOOPノードをつくる。
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

# 設定情報（cfgオブジェクト）を初期化し、複数のソースからの設定情報をマージして最終的な設定情報を出力する関数。
def setup_cfg(args):
    # デフォルト設定情報の作成
    cfg = get_cfg_default()
    # 新しい設定変数の追加
    extend_cfg(cfg)

    # 1. データセット設定ファイルからのマージ（結合）
    # merge_from_file()...与えられた情報を既存cfgオブジェクトにマージするメソッド
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. メソッド設定ファイルからのマージ
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. 直接指定されたコマンドライン引数からのマージ:
    reset_cfg(cfg, args)

    # 4. オプショナルなコマンドライン引数（直接指定された引数以外の引数が）からのマージ:
    cfg.merge_from_list(args.opts)

    # freeze()...CfgNodeオブジェクトを凍結する関数。設定情報の変更不可になる。
    cfg.freeze()

    return cfg


def main(args):
    # 設定のセットアップ
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        # 同じシードを用いたときに、pythonのrandomモジュールやPyTorchのCPU用ランダムシードなどを同じ状態からスタートし、実験結果を再現可能にする
        set_random_seed(cfg.SEED)
    # ログの出力ディレクトリやレベルなどの設定
    setup_logger(cfg.OUTPUT_DIR)

    # GPUの利用確認
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # コマンドライン引数と設定情報を表示
    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    # 設定情報に基づいたトレーナーオブジェクトの生成。モデルの訓練や評価を行う。
    trainer = build_trainer(cfg)

    # 評価
    if args.eval_only:
        # 指定されたディレクトリから保存されたモデルを読み込む
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    # 訓練
    if not args.no_train:
        trainer.train()


# __name__...特殊な変数。モジュールが直接実行された場合には "__main__" になり、モジュールが他のスクリプトからインポートされた場合にはモジュール名になる。
if __name__ == "__main__":
    # argparse...コマンドライン引数の定義や解析の設定をする為のライブラリ
    # ArgumentParser...argparsのクラス
    # parser...argparse.ArgumentParserクラスのインスタンス
    parser = argparse.ArgumentParser()
    # add_argument()...コマンドライン引数を定義して追加するメソッド
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    # parse_args()...コマンドライン引数を解析し、argsオブジェクトに格納。例えば、args.root は --root オプションに対応し、その値を取得。
    args = parser.parse_args()
    main(args)
