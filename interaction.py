'''
・ターミナルコマンド
  python interaction.py {model_path} {json_path} {init_ctx} {num_input_images}
  python interaction.py "output/oxford_flowers_a_flower_photo_of_a/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50" "./data/oxford_flowers/split_zhou_OxfordFlowers.json" "the flower photo of a" 1
  python interaction.py "output/oxford_pets/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50" "./data/oxford_pets/split_zhou_OxfordPets.json" "the animal pet photo of a" 1
  python interaction.py "output/caltech101_a_color_image_of_a/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50" "./data/caltech-101/split_zhou_Caltech101.json" "the color image of a" 1
  python interaction.py "output/caltech101_a_color_image_of_a/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50" "./data/caltech-101/split_zhou_Caltech101.json" "At the airport you can spot a" 1

・分類を行うデータセットに応じて書き換える文
  if len(tokenizer.encode(name)) == 2: 144行目
  PIL_image = Image.open(f'./data/oxford_flowers/jpg/{image_paths[0]}') 168行目
  PIL_image = Image.open(f'./data/oxford_flowers/jpg/{image_paths[i]}') 731行目

・model_path
  "output/oxford_flowers_random/CoOp/rn50_ep50_16shots/nctx5_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50"
  "output/caltech101_a_color_image_of_a/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50"
  "output/oxford_pets/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50"

・json_path
  "./data/oxford_flowers/split_zhou_OxfordFlowers.json"
  "./data/caltech-101/split_zhou_Caltech101.json"
  "./data/oxford_pets/split_zhou_OxfordPets.json"

・init_ctx
  "the flower photo of a"
  "the color image of a"
  "the animal pet photo of a"
'''

import os
import sys
import argparse
import numpy as np
import torch
from clip.simple_tokenizer import SimpleTokenizer
from clip import clip
import json
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import  itertools
import math
import matplotlib.pyplot as plt
import time
from matplotlib.colors import TwoSlopeNorm

torch.set_printoptions(edgeitems=1050)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# split_zhou_{dataset_name}.jsonからテスト画像へのパスをロードする関数
def load_image_paths(json_file_path, groundtruth_classname):
    # JSONファイルを開いてデータを読み込む
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # 画像パスを格納するリストを初期化
    image_paths = []

    # "train"キーに対応するデータから画像パスを抽出
    for item in data['test']:
        if item[2] == groundtruth_classname:
            image_path = item[0]  # 各リストの最初の要素が画像パス
            image_paths.append(image_path)
    return image_paths

# split_zhou_{dataset_name}.jsonから分類クラス名をロードする関数
def load_class_names(json_file_path):
    # JSONファイルを開いてデータを読み込む
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # クラス名を格納するセットを初期化（重複を避けるため）
    class_names = set()

    # "train"キーに対応するデータからクラス名を抽出
    for item in data['train']:
        class_name = item[2]  # 各リストの3番目の要素がクラス名
        class_names.add(class_name)
    
    # セットをリストに変換して返す（必要に応じてソートも可能）
    return sorted(list(class_names))

# cat_to_name.jsonから分類クラス名をロードする関数
def load_classnames(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # JSONデータからクラス名のみを取り出してリストに格納
    classnames = list(data.values())
    
    return classnames

def load_clip_to_gpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cuda").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cuda")

    model = clip.build_model(state_dict or model.state_dict())
    return model

# 画像の前処理用のクラスを定義する関数
def preprocess(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # clip_model.

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str, help="Path to the model file.")
parser.add_argument("json_path", type=str, help="Path to the JSON file.")
parser.add_argument("init_ctx", type=str, help="Initial value of context")
parser.add_argument("num_input_images", type=int, help="Number of input images")
args = parser.parse_args()
print(f"Model file path: {args.model_path}")
print(f"JSON file path: {args.json_path}")
print(f"Initial value of context: {args.init_ctx}")
print(f"Number of input images: {args.num_input_images}")

tokenizer = SimpleTokenizer()
clip_model = load_clip_to_gpu()
clip_model = clip_model.to("cuda")
transform = preprocess(clip_model.visual.input_resolution) # 前処理用のオブジェクト

# 指定したトークン数にBPEで分割されるクラスに絞る
json_file_path = args.json_path
classnames = load_class_names(json_file_path)
restrict_classnames = []
for name in classnames:
  if len(tokenizer.encode(name)) == 2:
      restrict_classnames.append(name)

print(f"Number of restricted classnames: {len(restrict_classnames)}")
print(f"List of restricted classnames: {restrict_classnames}")

# 学習されたトークンをロード
fpath = args.model_path
prompt_learner = torch.load(fpath, map_location="cuda")["state_dict"]
ctx = prompt_learner["ctx"].float()
ctx_length = ctx.size(0)
print(f"Number of context tokens: {ctx_length}")

shap_values_list2 = []
shap_values_list3 = []
# SHAP値をクラス数、画像数に対して平均
for ind, groundtruth_classname in enumerate(restrict_classnames):
    # if(groundtruth_classname!="airplane"):
    #     continue
    if(ind==1):
        break

    # groundtruth_classname="hibiscus" # 正解クラス
    print(f"----- groundtruth classname: {groundtruth_classname} -----")
    # テスト画像へのパス
    image_paths = load_image_paths(json_file_path, groundtruth_classname)
    # 入力画像（PIL形式に変換し、PIL画像を前処理したもの）
    # PIL_image = Image.open(f'./data/oxford_flowers/jpg/{image_paths[3]}')
    PIL_image = Image.open(f'./data/caltech-101/101_ObjectCategories/{image_paths[0]}')
    # PIL_image = Image.open(f'./data/oxford_pets/images/{image_paths[0]}')
    PIL_image.show()
    image_input = transform(PIL_image).unsqueeze(0).to("cuda")

    # region -------------init_ctxで画像分類-------------
    # 自然言語文のテキストプロンプト
    init_ctx = args.init_ctx
    # IDベクトルに変換
    text_inputs = torch.cat([clip.tokenize(f"{init_ctx} {c}.") for c in classnames]).to("cuda")

    # 入力画像とテキストプロンプトの特徴ベクトルをエンコーダから取得
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input) # Size = [1, 1024]
        text_features = clip_model.encode_text(text_inputs) # Size = [102, 1024] | 特徴ベクトル = encode_text（IDベクトル）
    image_features = image_features / image_features.norm(dim=-1, keepdim=True) #正規化
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 類似度を計算
    logit_scale = clip_model.logit_scale.exp() # logitのスケール値
    logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1) # Size = [1, 102]

    # top5の予測クラスと予測確率を表示
    values, indices = logits[0].topk(3)
    print("Zero-Shot-CLIPの予測クラス: 予測確率")
    for value, index in zip(values, indices):
        print(f"{classnames[index]:>16s}: {100 * value.item():.2f}%")

    debag_logits = (logit_scale * image_features @ text_features.t())
    values, indices = debag_logits[0].topk(1)
    # print(f"positive_logit = {values[0].item():.2f}")
    # endregion

    # region -------------ctxで画像分類------------------
    fpath = args.model_path             # モデルファイルパス
    prompt_learner = torch.load(fpath, map_location="cuda")["state_dict"]
    ctx = prompt_learner["ctx"].float() # Size = [5, 512]

    # 各クラスに対応する自然言語のテキストプロンプトを作成
    classnames = [name.replace("_", " ") for name in classnames]
    name_lens = [len(tokenizer.encode(name)) for name in classnames]
    prompts = [init_ctx + " " + name + "." for name in classnames]
    n_ctx = len(init_ctx.split(" "))
    # 作成したプロンプトからIDベクトル、埋め込みベクトルを取得
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to("cuda")
    with torch.no_grad():
        embeddings = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype) # Size = [102, 77, 512]
    # ctx以外のトークンの埋め込みベクトルを取得
    prefix_token = embeddings[:, :1, :]         # Size = [102, 1, 512]  | SOSトークンの特徴ベクトル
    suffix_token = embeddings[:, 1 + n_ctx:, :] # Size = [102, 71, 512] | CLSトークン, 「.」トークン, EOSトークン, それ以降のトークンの特徴ベクトル

    # 「SOS + ctx + {classname} + . + EOS」のトークン埋め込みベクトルを構築
    n_cls = len(classnames)
    ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1) # Size = [102, 5, 512]
    prompts_embeddings = torch.cat(
        [
            prefix_token,  # [n_cls, 1, embedding_dim]
            ctx,           # [n_cls, n_ctx, embedding_dim]
            suffix_token,  # [n_cls, *, embedding_dim]
        ],
        dim=1,
    ).type(clip_model.dtype) # Size = [102, 77, 512]
    # print(f"正解クラスのprompts_embeddings = {prompts_embeddings[100, :13, :5]}") # OK

    # テキスト特徴ベクトルを取得
    x = prompts_embeddings + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)                                  # NLD -> LND バッチサイズ（N）、シーケンス（コンテキスト）長（L）、特徴次元（D）
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)                                  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)       # Size = [102, 77, 512] = [n_cls, 1 + n_ctx + *, transformer.width]
    # EOSトークンの特徴ベクトルを取り出す | テキスト射影でテキスト特徴ベクトルを画像特徴ベクトルと同じ次元に変換
    text_features = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection # Size = [102, 1024]
    # print(text_features[100, :10])

    # 入力画像の特徴ベクトルをエンコーダから取得
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input) # Size = [1, 1024]
    image_features = image_features / image_features.norm(dim=-1, keepdim=True) #正規化
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 類似度を計算
    logit_scale = clip_model.logit_scale.exp() # logitのスケール値
    logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1) # Size = [1, 102]

    # top5の予測クラスと予測確率を表示
    values, indices = logits[0].topk(3)
    print("CLIP+CoOpの予測クラス: 予測確率")
    for value, index in zip(values, indices):
        print(f"{classnames[index]:>16s}: {100 * value.item():.2f}%")

    debag_logits = (logit_scale * image_features @ text_features.t())
    values, indices = debag_logits[0].topk(1)
    # print(f"positive_logit = {values[0].item():.2f}")
    # endregion

    # region -------------init_ctxをSHAP分析-------------
    # テキストプロンプトのトークン化関数
    def my_tokenize(texts, tokenizer, max_length=77):
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = [tokenizer.encode(text) for text in texts]
        result = torch.zeros((len(all_tokens), max_length), dtype=torch.long)
        for i, tokens in enumerate(all_tokens):
            length = min(len(tokens), max_length)
            result[i, :length] = torch.tensor(tokens[:length])
        return result

    # 一つの背景データを処理する予測関数（入力形式は単一のテンソル）
    def model_predict(text_input_tensor, PIL_image):
        # 入力画像（PIL形式に変換し、PIL画像を前処理したもの）
        image_input = transform(PIL_image).unsqueeze(0).to("cuda")

        # 入力画像とテキストプロンプトの特徴ベクトルをエンコーダから取得
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input) # Size = [1, 1024]
            text_features = clip_model.encode_text(text_input_tensor) # Size = [1, 1024] | 特徴ベクトル = encode_text（IDベクトル）
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 類似度を計算
        logit_scale = clip_model.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1) # Size = [1, 102]

        # top5の予測クラスと予測確率を表示
        values, indices = logits[0].topk(5)
        # print("Zero-Shot-CLIPの予測クラス: 予測確率")
        # for value, index in zip(values, indices):
        #     print(f"{classnames[index]:>16s}: {100 * value.item():.2f}%")
        
        # 正解ペアの類似度を返す ※最大類似度のペアが正解ペアの類似度か確認が必要
        return values[0].cpu().detach().numpy()

    # 複数の背景データをまとめて処理する予測関数（入力形式はテンソルのバッチ）
    def batch_model_predict(batch_text_inputs_tensor, PIL_image): # batch_text_inputs_tensor Size = [batchsize=1024, 77]
        batch_size = batch_text_inputs_tensor.size(0)
        outputs = []

        # バッチ処理
        for i in range(batch_size):
            text_input_tensor = batch_text_inputs_tensor[i].unsqueeze(0)  # Size = [1, 77] | バッチのi番目のテンソルを取得

            # 以下の処理は元のmodel_predict関数と同じ
            image_input = transform(PIL_image).unsqueeze(0).to("cuda")

            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)      # Size = [1, 1024]
                text_features = clip_model.encode_text(text_input_tensor) # Size = [1, 1024]
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logit_scale = clip_model.logit_scale.exp() # 100
                logits = (logit_scale * image_features @ text_features.t()) # Size = [1, 1] | 予測値が必ず1になってしまうため、softmaxはなし | -100〜100の範囲で分布
                # logits = (image_features @ text_features.t()) # -1〜1の範囲で分布

                # 出力値を保存
                outputs.append(logits[0].cpu().detach().numpy()) # len = [num_sumples=100]

        return np.array(outputs)

    # （ID=0, 490406, 49407ではない）有効トークンの全組合わせから、それぞれの背景データ生成関数
    def generate_background_data(text_input, num_samples=1024, pad_token_id=0, sos_token_id = 49406, eos_token_id = 49407):
        # print(text_input)
        # print(len(text_input))
        max_length = text_input.size(1)
        # 有効なトークンのtext_inputにおけるインデックスを取得
        # valid_indices = torch.arange(max_length, device=text_input.device)[text_input[0] != pad_token_id]
        valid_indices = torch.arange(max_length, device=text_input.device)[(text_input[0] != pad_token_id)&(text_input[0] != sos_token_id)&(text_input[0] != eos_token_id)]
        # print(f"有効なトークンのインデックス: {valid_indices}")
        valid_indices = valid_indices.cpu().numpy()

        # 有効なトークンのインデックスのすべての組み合わせを生成
        all_combinations = list(itertools.combinations(valid_indices, r) for r in range(1, len(valid_indices) + 1))
        all_combinations = list(itertools.chain.from_iterable(all_combinations))
        
        # 組み合わせの数が num_samples を超える場合は、先頭の num_samples のみを使用
        selected_combinations = all_combinations[:num_samples]
        # print(selected_combinations)

        background_data = []
        for combination in selected_combinations:
            # 指定されたインデックスのトークンを保持し、他を取り除く
            mask = torch.zeros(max_length, dtype=torch.bool, device=text_input.device)
            mask[list(combination)] = True
            remaining_tokens = text_input[0, mask]

            # SOSとEOSトークンを追加
            remaining_tokens = torch.cat([
                torch.tensor([sos_token_id], device=text_input.device),
                remaining_tokens,
                torch.tensor([eos_token_id], device=text_input.device)
            ])

            # 取り除かれたトークンの後にパディングを挿入
            num_pads = max_length - len(remaining_tokens)
            padded_tokens = torch.cat([remaining_tokens, torch.full((num_pads,), pad_token_id, device=text_input.device)])
            background_data.append(padded_tokens.unsqueeze(0))

        # [49406, 49407, 0, ..., 0]を背景データに含める
        background_data_tensor = torch.cat(background_data)
        background_data_of_sos_eos = torch.tensor([[sos_token_id, eos_token_id] + [pad_token_id] * (max_length - 2)], device=text_input.device)
        background_data = torch.cat([background_data_of_sos_eos, background_data_tensor], dim=0)
        
        return background_data, selected_combinations

    # テキストプロンプトの準備（IDベクトル）
    # groundtruth_classname = "passion flower"  # 入力画像の正解ペアのクラス名
    # groundtruth_classname = "face"
    # groundtruth_classname = "airplane"
    # groundtruth_classname = "Russian Blue"
    # groundtruth_classname = "rose"
    text_input = clip.tokenize(f"{init_ctx} {groundtruth_classname}.").to("cuda") # [1, 77]
    prompt_length = text_input[0].argmin(dim=-1).item()
    # text_prompt = f"a flower photo of a {groundtruth_classname}."
    # text_input = my_tokenize(text_prompt, tokenizer).to("cuda")
    # print(f"text_input = {text_input} {text_input.shape}")

    # 背景データの生成
    background_data, combinations = generate_background_data(text_input) # Size = [2^(len(valid_indices)) - 1, 77]
    # print(f"background_data = {background_data[:, :10]} {background_data.shape}")
    # print(f"combinations = {combinations} {len(combinations)}")

    # region 各背景データでの予測値を計算
    # background_data_predictions = batch_model_predict(background_data, PIL_image)
    # print(f"background_data_predictions = {background_data_predictions} {len(background_data_predictions)}")
    # endregion

    # 引数のテキストプロンプトの各トークンのSHAP値を計算する関数
    def compute_shap_values(background_data, model_predict, text_input, PIL_image):
        # 各トークンのSHAP値を格納する配列を初期化
        shap_values = np.zeros(text_input.size(1)) # len = 77
        # 予測貢献度を求めるトークンのid
        shap_ids = text_input[(text_input != 49406) & (text_input != 49407) & (text_input != 0)]
        # print(f"SHAP値を計算する各トークンのid = {shap_ids} {shap_ids.shape}")
        # |N|
        N_elenum = shap_ids.shape[0]
        # print(f"N_elenum = {N_elenum}")

        # i(=shap_id)のトークンの予測貢献度を求める
        for index, i in enumerate(shap_ids):
            # print(f"--------------元のプロンプトの{index}個目のid={i.item()}のトークンの予測貢献度を計算--------------")
            # 集合Sかつ{i}を作成
            S_and_i = []
            for data in background_data:
                if i in data:
                    S_and_i.append(data.unsqueeze(0)) # iが含まれるトークン列をS_and_iに追加
            S_and_i = torch.cat(S_and_i, dim=0)       # Size = [64, 77]

            # Sかつ{i}とSのトークン列での差分を計算して合計
            for Si in S_and_i: # Si Size = [77]
                # 集合Sかつ{i}のトークン列Si
                # 集合Sのトークン列
                S = Si[Si != i]                                               # iをSiから省く
                S = torch.cat((S, torch.tensor([0], device='cuda:0')), dim=0) # padding_id=0を追加
                # f(Sかつ{i})
                f_Si = model_predict(Si.unsqueeze(0), PIL_image).item()
                # f(S)
                f_S = model_predict(S.unsqueeze(0), PIL_image).item()
                # f(Si) - f(S)
                temp = f_Si - f_S
                # P(S|N/{i})
                S_elenum = S[S != 0].shape[0] - 2 # |S|
                P = (math.factorial(S_elenum) * math.factorial(N_elenum - S_elenum - 1)) / math.factorial(N_elenum)
                # 限界貢献度を合計し、予測貢献度を算出
                shap_values[index+1] += P * temp
                # debag
                # print(f"Si = {Si[:10]} {Si.shape}")
                # print(f"S = {S[:10]} {S.shape}")
                # print(f"f_Si = {f_Si}")
                # print(f"f_S = {f_S} ")
                # print(f"f_Si - f_S = {temp} ")
                # print(f"S_elenum = {S_elenum} ")
                # print(f"P = {P} ")
                # print(f"P(S|N/{i})f(Si) - f(S) = {P * temp} ")

        return shap_values

    # 引数のテキストプロンプトの各トークン同士のinteractionを計算する関数
    def compute_interactions(background_data, model_predict, text_input, PIL_image):
        # 予測貢献度を求めるトークンのid
        shap_ids = text_input[(text_input != 49406) & (text_input != 49407) & (text_input != 0)]
        # print(f"SHAP値を計算する各トークンのid = {shap_ids} {shap_ids.shape}")
        # |N|
        N_elenum = shap_ids.shape[0]
        # 各interactionを格納する配列
        interactions = np.zeros((N_elenum, N_elenum))

        # iとjのinteractionを計算する
        for index, i in enumerate(shap_ids):
            for index2, j in enumerate(shap_ids):
                if index == index2:
                    continue

                # φ({i, j}|N')
                # 集合Sかつ{i, j}を作成
                S_and_i_j = []
                for data in background_data:
                    if i in data and j in data:
                        S_and_i_j.append(data.unsqueeze(0)) # {i, j}が含まれるトークン列をS_and_i_jに追加
                S_and_i_j = torch.cat(S_and_i_j, dim=0)     # Size = [32, 77]

                # S\{i, j}についてループ
                for Sij in S_and_i_j: # Sij Size = [77]
                    # 集合Sかつ{i}のトークン列Si
                    Si = Sij[Sij != j] # jをSijから省く
                    Si = torch.cat((Si, torch.tensor([0], device='cuda:0')), dim=0) # padding_id=0を追加
                    # 集合Sかつ{j}のトークン列Si
                    Sj = Sij[Sij != i] # iをSijから省く
                    Sj = torch.cat((Sj, torch.tensor([0], device='cuda:0')), dim=0)
                    # 集合Sのトークン列
                    S = Sj[Sj != j]    # jをSjから省く
                    S = torch.cat((S, torch.tensor([0], device='cuda:0')), dim=0)
                    # f(Sかつ{i, j})
                    f_Sij = model_predict(Sij.unsqueeze(0), PIL_image).item()
                    # f(Sかつ{i})
                    f_Si = model_predict(Si.unsqueeze(0), PIL_image).item()
                    # f(Sかつ{j})
                    f_Sj = model_predict(Sj.unsqueeze(0), PIL_image).item()
                    # f(S)
                    f_S = model_predict(S.unsqueeze(0), PIL_image).item()
                    # f(Sかつ{i, j}) - f(Sかつ{i}) - f(Sかつ{j}) + f(S)
                    temp = f_Sij - f_Si - f_Sj + f_S
                    # P(S|N/{i})
                    S_elenum = S[S != 0].shape[0] - 2 # |S|
                    P = (math.factorial(S_elenum) * math.factorial(N_elenum - S_elenum - 2)) / (2 * math.factorial(N_elenum - 1))
                    # 限界貢献度を合計し、予測貢献度を算出
                    interactions[index, index2] += P * temp
        return interactions
    
    # # 各トークンのinteractionを計算
    # interactions = compute_interactions(background_data, batch_model_predict, text_input, PIL_image)
    # for row in interactions:
    #     print(" ".join("{:>6.1f}".format(x) for x in row))

    # # 各トークンのSHAP値を表示
    # print("\n初期値のプロンプトの各トークンのShapley Value")
    # for i, shap_value in enumerate(shap_values):
    #     token_id = text_input[0, i].item()
    #     if token_id != 0 and token_id != 49406 and token_id != 49407:  # ID=0以外のトークンのSHAP値を表示 ※ID=0のトークンのSHAP値はそもそも計算していない
    #         token = tokenizer.decode([token_id])
    #         print(f"Token: {token:<15} Shapley Value: {shap_value:.4f}")  
    # endregion

    # region -------------ctxをSHAP分析------------------
    def count_valid_tokens(prompt_embedding):
        embedding = clip_model.token_embedding
        pad_token_embedding = embedding(torch.tensor([0]).to("cuda"))     # Size = [1, 512]
        sos_token_embedding = embedding(torch.tensor([49406]).to("cuda")) # Size = [1, 512]
        eos_token_embedding = embedding(torch.tensor([49407]).to("cuda")) # Size = [1, 512]

        # prompt_embeddingから各トークンに対して、pad/sos/eosトークンとの等価性をチェック
        is_pad = torch.all(prompt_embedding == pad_token_embedding, dim=-1)
        is_sos = torch.all(prompt_embedding == sos_token_embedding, dim=-1)
        is_eos = torch.all(prompt_embedding == eos_token_embedding, dim=-1)

        # 有効なトークン（pad, sos, eos以外）のマスクを作成
        valid_mask = ~(is_pad | is_sos | is_eos)

        # 有効なトークンの数をカウント
        num_valid_tokens = valid_mask.sum().item()

        return num_valid_tokens

    # （ID=0, 49406, 49407の埋め込みベクトルではない）有効トークンの全組合わせから、それぞれの背景データ生成関数
    def generate_ctx_background_datas(prompt_embedding, num_samples=1024):
        embedding = clip_model.token_embedding
        pad_token_embedding = embedding(torch.tensor([0]).to("cuda"))     # Size = [1, 512]
        sos_token_embedding = embedding(torch.tensor([49406]).to("cuda")) # Size = [1, 512]
        eos_token_embedding = embedding(torch.tensor([49407]).to("cuda")) # Size = [1, 512]
        # print(f"pad_token_embeddings = {pad_token_embedding[0, :5]}")
        # print(f"sos_token_embeddings = {sos_token_embedding[0, :5]}")
        # print(f"eos_token_embeddings = {eos_token_embedding[0, :5]}")

        num_valid_tokens = count_valid_tokens(prompt_embedding)

        max_length = prompt_embedding.size(1) # prompt_embeddingのトークン数
        valid_indices = torch.arange(num_valid_tokens, device=prompt_embedding.device).cpu().numpy()
        valid_indices = [x + 1 for x in valid_indices] # 有効なトークンのインデックス
        # print(f"valid_indices = {valid_indices}")

        # 有効なトークンのインデックスのすべての組み合わせを生成
        all_combinations = list(itertools.combinations(valid_indices, r) for r in range(1, len(valid_indices) + 1))
        all_combinations = list(itertools.chain.from_iterable(all_combinations))
        # 組み合わせの数が num_samples を超える場合は、先頭の num_samples のみを使用
        selected_combinations = all_combinations[:num_samples]
        # print(selected_combinations)

        prompt_embedding_background_datas = []
        for combination in selected_combinations:
            # combinationで指定されたインデックスのprompt_embeddingのトークンを保持し、他を取り除く
            mask = torch.zeros(max_length, dtype=torch.bool, device=prompt_embedding.device)
            mask[list(combination)] = True
            remaining_tokens = prompt_embedding[:, mask, :]

            # EOS, SOSの埋め込みベクトルを追加
            remaining_tokens_with_sos_eos = torch.cat([
            sos_token_embedding.unsqueeze(0),  # SOSの埋め込みベクトルを追加（バッチ次元を保持するためunsqueeze(0)を使用）
            remaining_tokens,
            eos_token_embedding.unsqueeze(0)   # EOSの埋め込みベクトルを追加（バッチ次元を保持するためunsqueeze(0)を使用）
            ], dim=1)

            # パディングトークンの埋め込みベクトルを取得（ID=0のトークン）
            pad_token_embedding = clip_model.token_embedding(torch.tensor([[0]], device=prompt_embedding.device))
            # 取り除かれたトークンの後にパディングを挿入
            num_paddingss = max_length - len(combination) - 2
            padded_tokens = torch.cat([remaining_tokens_with_sos_eos, pad_token_embedding.repeat(1, num_paddingss, 1)], dim=1)
            prompt_embedding_background_datas.append(padded_tokens)

        # [49406, 49407, 0, ..., 0]の埋め込みベクトルを取得
        empty_prompt_ids = [49406, 49407] + [0] * 75
        empty_prompt_ids_tensor = torch.tensor(empty_prompt_ids).unsqueeze(0).to("cuda")
        empty_prompt_embedding = embedding(empty_prompt_ids_tensor) # Size = [1, 77, 512]
        # print(empty_prompt_embedding[0, :5, :5])

        # [49406, 49407, 0, ..., 0]の埋め込みベクトルを背景データに含める
        prompt_embedding_background_datas = torch.cat(prompt_embedding_background_datas, dim=0)
        prompt_embedding_background_datas = torch.cat([empty_prompt_embedding, prompt_embedding_background_datas], dim=0) # Size = [256(=2^len(valid_indices)), 77, 512]
        # print(prompt_embedding_background_datas[255, :11, :5])

        return prompt_embedding_background_datas, selected_combinations

    # 一つの背景データを処理する予測関数（入力にEOSのインデックス）
    def ctx_model_predict_len_combination(ctx_background_data, len_combination, PIL_image):
        # 入力画像（PIL形式に変換し、PIL画像を前処理したもの）
        image_input = transform(PIL_image).unsqueeze(0).to("cuda")

        # バッチ次元を追加（2→3）し、text encoderの入力形式と合わせる
        ctx_background_data = ctx_background_data.unsqueeze(0).type(clip_model.dtype) # [1, 77, 512]
        # print(f"ctx_background_data = {ctx_background_data[:, :13, :5]} {ctx_background_data.shape}")

        # テキスト特徴ベクトルを取得
        x = ctx_background_data + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)                             
        x = clip_model.ln_final(x).type(clip_model.dtype) # Size = [1, 77, 512] = [n_cls, 1 + n_ctx + *, transformer.width]
        # EOSトークンの特徴ベクトルを取り出す | テキスト射影でテキスト特徴ベクトルを画像特徴ベクトルと同じ次元に変換
        text_features = x[:, len_combination-1, :] @ clip_model.text_projection  # Size = [1, 1024] | 最後のトークン（通常はEOS）を使用
        # text_features = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ clip_model.text_projection # Size = [1, 1024]
        # print(f"text_features = {text_features[0, :10]} {text_features.shape}")

        # 入力画像の特徴ベクトルをエンコーダから取得
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input) # Size = [1, 1024]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) #正規化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 類似度を計算
        logit_scale = clip_model.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t()) # Size = [1, 1] | 予測値が必ず1になってしまうため、softmaxはなし

        # 正解ペアの類似度を返す ※最大類似度のペアが正解ペアの類似度か確認が必要
        return logits[0].cpu().detach().numpy()

    # 一つの背景データを処理する予測関数
    def ctx_model_predict(ctx_background_data, PIL_image):
        # 入力画像の準備
        image_input = transform(PIL_image).unsqueeze(0).to("cuda")

        # バッチ次元を追加し、Text Encoderの入力形式に合わせる
        ctx_background_data = ctx_background_data.unsqueeze(0).type(clip_model.dtype)  # [1, 77, 512]

        # EOSトークンの埋め込みベクトルを取得
        eos_embedding = clip_model.token_embedding(torch.tensor([49407], device='cuda')).type(clip_model.dtype) # [1, 1, 77]

        # ctx_background_dataの各トークンとEOSトークンの埋め込みベクトルの類似度を計算
        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        similarity = cos(ctx_background_data, eos_embedding) # [1, 77]
        # print(f"similarity = {similarity} {similarity.shape}") # ID=49407→パディングID=0の順にEOSとの類似度が高い| OK
        # 組み合わせを決める際、SOS,EOSは必ず含める方がいいかも。SHAP値もSOS, EOSは計算しない。

        # 最大類似度を持つトークンの位置を見つける
        _, eos_index = torch.max(similarity, dim=1)
        eos_index = eos_index.item()
        # print(eos_index) OK

        # テキスト特徴ベクトルの取得
        x = ctx_background_data + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x).type(clip_model.dtype)

        # EOSの位置に基づいて特徴ベクトルを取り出す
        text_features = x[0, eos_index, :] @ clip_model.text_projection  # Size = [1024]

        # 画像特徴ベクトルの取得と正規化
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 類似度の計算
        logit_scale = clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.unsqueeze(-1)  # Size = [1]

        return logits.item()

    n_cls = 1 # バッチ数
    # 正解クラスに対応する自然言語のテキストプロンプトを作成
    prompt = [init_ctx + " " + groundtruth_classname + "."]      # list | len = 1
    n_ctx = len(init_ctx.split(" "))                             # len = 5
    n_class_token = len(tokenizer.encode(groundtruth_classname)) # len = 2
    prompt_length = 1 + n_ctx + n_class_token + 1 + 1            # len = 10 | SOS ini_ctx CLASS . EOS
    # 作成したプロンプトからIDベクトル、埋め込みベクトルを取得
    tokenized_prompt = torch.cat([clip.tokenize(prompt)]).to("cuda")   # Size = [1, 77]
    with torch.no_grad():
        embedding = clip_model.token_embedding(tokenized_prompt).type(clip_model.dtype) # Size = [n_cls=1, 77, 512]
    # ctx以外のトークンの埋め込みベクトルを取得
    prefix_token = embedding[:, :1, :]           # Size = [1, 1, 512]  | SOSトークンの特徴ベクトル
    suffix_token = embedding[:, 1 + n_ctx:, :]   # Size = [1, 71, 512] | CLSトークン, 「.」トークン, EOSトークン, それ以降のトークンの特徴ベクトル

    ctx = prompt_learner["ctx"].float()          # Size = [5, 512]
    # 「SOS + ctx + {classname} + . + EOS」のトークン埋め込みベクトルを構築
    ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1) # Size = [n_cls=1, 5, 512]
    prompt_embedding = torch.cat(
        [
            prefix_token,    # [n_cls=1, 1, embedding_dim]
            ctx,             # [n_cls=1, n_ctx, embedding_dim]
            suffix_token,    # [n_cls=1, *, embedding_dim]
        ],
        dim=1,
    ).type(clip_model.dtype) # Size = [1, 77, 512]
    # print(f"prompt_embedding = {prompt_embedding[0, :13, :5]}") # OK

    # 背景データの生成
    ctx_background_datas, combinations = generate_ctx_background_datas(prompt_embedding) # Size = [1023, 77, 512]
    # print(f"background_data = {ctx_background_datas[1022, :13, :5].type(clip_model.dtype)} {ctx_background_datas.shape}")# OK

    # 引数のテキストプロンプトの各トークンのSHAP値を計算する関数
    def compute_ctx_shap_values(background_data, model_predict, prompt_embedding, PIL_image):
        # 各トークンのSHAP値を格納する配列を初期化
        shap_values = np.zeros(prompt_embedding.size(1)) # len = 77
        # 予測貢献度を求めるトークンのembedding
        embedding = clip_model.token_embedding
        num_valid_tokens = count_valid_tokens(prompt_embedding) # len = 8
        # print(f"num_valid_tokens={num_valid_tokens}")
        shap_embedding = prompt_embedding[:, 1:num_valid_tokens+1, :] # Size = [1, 8(=len(num_valid_tokens)), 512]
        # print(f"SHAP値を計算する各トークンのembedding = {shap_embedding[0, :, :5]} {shap_embedding.shape}")
        # |N|
        N_elenum = num_valid_tokens
        # print(f"N_elenum = {N_elenum}")

        # i(=shap_embedding)のトークンの予測貢献度を求める
        # for index, i in enumerate(shap_embedding):
        for index in range(1, N_elenum+1):
            i = prompt_embedding[:, index, :] # Size = [1, 512]
            # print(f"--------------元のプロンプトの{index}個目のembedding={i[:,0].item():5f}のトークンの予測貢献度を計算--------------")
            # print(f"SHAP値を求めるトークンiのembedding = {i[:,:5]} {i.shape}")
            # 集合Sかつ{i}を作成
            S_and_i = []
            for data in background_data: # data Size = [77, 512] | background_data Size = [256, 77, 512]
                if torch.any(torch.all(data == i, dim=1)):
                    S_and_i.append(data.unsqueeze(0)) # iが含まれるトークン列をS_and_iに追加
                # if i in data:
                #     S_and_i.append(data.unsqueeze(0)) # iが含まれるトークン列をS_and_iに追加
            S_and_i = torch.cat(S_and_i, dim=0)       # Size = [128, 77, 512]

            # print(f"S_and_i[0,:,:5] = {S_and_i[0,:,:5]} {S_and_i[0].shape}")
            # print(f"S_and_i.shape = {S_and_i.shape}")

            # Si = S_and_i[0,:,:]
            # S = Si[~torch.all(Si == i, dim=1)]
            # print(S[:5,:5])
            # print(S.shape)

            # Sかつ{i}とSのトークン列での差分を計算して合計
            for Si in S_and_i: # Si Size = [77, 512] | S_and_i Size = [128, 77, 512]            
                # 集合Sかつ{i}のトークン列Si
                # 集合Sのトークン列
                S = Si[~torch.all(Si == i, dim=1)]                            # Size = [76, 512] | iをSiから省く
                pad_token_embedding = embedding(torch.tensor([0]).to("cuda")) # Size = [1, 512]
                S = torch.cat((S, pad_token_embedding), dim=0) # padding_token_embeddingを追加 | Size = [77, 512]
                # f(Sかつ{i})
                f_Si = model_predict(Si, PIL_image)
                # f(S)
                f_S = model_predict(S, PIL_image)
                # f(Si) - f(S)
                temp = f_Si - f_S
                # P(S|N/{i})
                S_elenum = count_valid_tokens(S.unsqueeze(0))
                P = (math.factorial(S_elenum) * math.factorial(N_elenum - S_elenum - 1)) / math.factorial(N_elenum)
                # 限界貢献度を合計し、予測貢献度を算出
                shap_values[index] += P * temp

                # debag
                # print(f"Si = {Si[:10]} {Si.shape}")
                # print(f"S = {S[:10]} {S.shape}")
                # print(f"f_Si = {f_Si}")
                # print(f"f_S = {f_S} ")
                # print(f"f_Si - f_S = {temp} ")
                # print(f"S_elenum = {S_elenum} ")
                # print(f"P = {P} ")
                # print(f"P(S|N/{i})f(Si) - f(S) = {P * temp} ")

        return shap_values

    # 引数のテキストプロンプトの各トークン同士のinteractionを計算する関数
    def compute_ctx_interactions(background_data, model_predict, prompt_embedding, PIL_image):
        embedding = clip_model.token_embedding          # 埋め込み表現
        N_elenum = count_valid_tokens(prompt_embedding) # |N|
        # 各interactionを格納する配列
        interactions = np.zeros((N_elenum+1, N_elenum+1))

        # i, j(=embedding)のトークン同士のinteractionを求める
        for index in range(1, N_elenum+1):
            for index2 in range(1, N_elenum+1):
                if index == index2:
                    continue

                i = prompt_embedding[:, index, :] # Size = [1, 512]
                j = prompt_embedding[:, index2, :] # Size = [1, 512]
                # 集合Sかつ{i, j}を作成
                S_and_i_j = []
                for data in background_data: # data Size = [77, 512] | background_data Size = [256, 77, 512]
                    if torch.any(torch.all(data == i, dim=1)) and torch.any(torch.all(data == j, dim=1)):
                        S_and_i_j.append(data.unsqueeze(0))   # i, jが含まれるトークン列をS_and_i_jに追加
                S_and_i_j = torch.cat(S_and_i_j, dim=0)       # Size = [128, 77, 512]

                # S\{i, j}についてループ
                for Sij in S_and_i_j: # Sij Size = [77, 512] | S_and_i_j Size = [128, 77, 512]            
                    # 集合Sかつ{i,j}のトークン列Sij
                    # 集合Sのトークン列
                    Sj = Sij[~torch.all(Sij == i, dim=1)]                         # Size = [76, 512] | iをSijから省く
                    Si = Sij[~torch.all(Sij == j, dim=1)]                         # Size = [76, 512] | jをSijから省く
                    pad_token_embedding = embedding(torch.tensor([0]).to("cuda")) # Size = [1, 512]
                    Sj = torch.cat((Sj, pad_token_embedding), dim=0) # padding_token_embeddingを追加 | Size = [77, 512]
                    Si = torch.cat((Si, pad_token_embedding), dim=0) # padding_token_embeddingを追加 | Size = [77, 512]
                    S = Si[~torch.all(Si == i, dim=1)]                            # Size = [76, 512] | iをSiから省く
                    S = torch.cat((S, pad_token_embedding), dim=0)   # padding_token_embeddingを追加 | Size = [77, 512]
                    # f(Sかつ{i, j})
                    f_Sij = model_predict(Sij, PIL_image)
                    # f(Sかつ{i})
                    f_Si = model_predict(Si, PIL_image)
                    # f(Sかつ{j})
                    f_Sj = model_predict(Sj, PIL_image)
                    # f(S)
                    f_S = model_predict(S, PIL_image)
                    # f(Si) - f(S)
                    temp = f_Sij - f_Si - f_Sj + f_S
                    # P(S|N/{i})
                    S_elenum = count_valid_tokens(S.unsqueeze(0))
                    P = (math.factorial(S_elenum) * math.factorial(N_elenum - S_elenum - 2)) / (2 * math.factorial(N_elenum - 1))
                    # 限界貢献度を合計し、予測貢献度を算出
                    interactions[index, index2] += P * temp

        return interactions
    
    # # 各トークンのinteractionを計算
    # interactions = compute_ctx_interactions(ctx_background_datas, ctx_model_predict, prompt_embedding, PIL_image)
    # interactions = interactions[1:, 1:]
    # for row in interactions:
    #     print(" ".join("{:>6.1f}".format(x) for x in row))

    '''
    # 各トークンのSHAP値を計算
    shap_values = compute_ctx_shap_values(ctx_background_datas, ctx_model_predict, prompt_embedding, PIL_image)

    # 各トークンのSHAP値を表示
    print("\n学習されたプロンプトの各トークンのSHAP値")
    for i, shap_value in enumerate(shap_values):
        if  i < prompt_length:  # EOSまでのトークンのSHAP値を表示
            if i != 0 and i != prompt_length - 1:
                print(f"trained Token{i} SHAP Value: {shap_value:.5f}")
    '''
    # endregion

    # region -------------init_ctxをSHAP分析（複数枚の画像に対するSHAP値の平均をとる）--------------
    # 入力画像のリスト（PIL画像形式）
    PIL_image_list = []
    num_input_images = args.num_input_images
    for i in range(num_input_images):
        # PIL_image = Image.open(f'./data/oxford_flowers/jpg/{image_paths[i]}')
        PIL_image = Image.open(f'./data/caltech-101/101_ObjectCategories/{image_paths[i]}')
        # PIL_image = Image.open(f'./data/oxford_pets/images/{image_paths[i]}')
        PIL_image_list.append(PIL_image) # len = 10

     # 各入力画像での各トークン同士のinteractionsを格納するリスト
    interactions_list = []
    for i, PIL_image in enumerate(PIL_image_list):
        print(f"{i+1}個目の入力画像に対する各トークンのinteractionを計算")
        start_time = time.time()  # 処理開始前の時刻
        interactions = compute_interactions(background_data, batch_model_predict, text_input, PIL_image)
        end_time = time.time()  # 処理終了後の時刻
        print(f"実行にかかった時間: {end_time - start_time}秒")
        interactions_list.append(interactions) # len = 10
        for row in interactions:
            print(" ".join("{:>6.1f}".format(x) for x in row))
        
    # NumPy配列に変換
    interactions_array = np.array(interactions_list)
    # 平均を計算
    average_interactions = np.mean(interactions_array, axis=0)
    print("平均Interactions:")
    for row in average_interactions:
        print(" ".join("{:>6.1f}".format(x) for x in row))

    # ヒートマップを生成
    norm = TwoSlopeNorm(vmin=average_interactions.min(), vcenter=0, vmax=average_interactions.max())
    plt.imshow(average_interactions, cmap='coolwarm', norm=norm, interpolation='nearest')
    plt.colorbar()  # カラーバーを追加
    plt.title('Heatmap of the Matrix')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    # ヒートマップをファイルに保存
    plt.savefig('initctx_interactions_heatmap.png', format='png')  # PNG形式で保存
    # リソースのクリーンアップ
    plt.close()

    # # 各入力画像での各トークンのSHAP値を格納するリスト
    # shap_values_list = []
    # for i, PIL_image in enumerate(PIL_image_list):
    #     print(f"{i+1}個目の入力画像に対する各トークンのSHAP値を計算")
    #     shap_values = compute_shap_values(background_data, batch_model_predict, text_input, PIL_image)
    #     shap_values_list.append(shap_values) # len = 10

    # ave_shap_values = 0
    # shap_values_list_array = np.array(shap_values_list) # [3, 77]
    # # print(shap_values_list_array[:, :12])
    # ave_shap_values = shap_values_list_array.sum(axis=0, keepdims=True)
    # # print(ave_shap_values[0, :12])
    # ave_shap_values /= len(shap_values_list) # [1, 77]
    # # print(ave_shap_values[0, :12])

    # shap_values_list2.append(ave_shap_values[0])

    # print("初期値のプロンプトの各トークンのSHAP値（各画像に対する平均）")
    # for i, ave_shap_value in enumerate(ave_shap_values[0]):
    #     if(shap_values[i] != 0):
    #         token_id = text_input[0, i].item()
    #         token = tokenizer.decode([token_id])
    #         print(f"{token:<15} SHAP値: {ave_shap_value:.2f}")
    # endregion

    # region -------------ctxをSHAP分析（複数枚の画像に対するSHAP値の平均をとる）-------------------
    # 各入力画像での各トークン同士のinteractionsを格納するリスト
    interactions_list = []
    for i, PIL_image in enumerate(PIL_image_list):
        print(f"{i+1}個目の入力画像に対する各トークン同士のinteractinを計算")
        interactions = compute_ctx_interactions(ctx_background_datas, ctx_model_predict, prompt_embedding, PIL_image)
        interactions = interactions[1:, 1:]
        interactions_list.append(interactions) # len = 10
        for row in interactions:
            print(" ".join("{:>6.1f}".format(x) for x in row))

    # NumPy配列に変換
    interactions_array = np.array(interactions_list)
    # 平均を計算
    average_interactions = np.mean(interactions_array, axis=0)
    print("平均Interactions:")
    for row in average_interactions:
        print(" ".join("{:>6.1f}".format(x) for x in row))
    
    # ヒートマップを生成
    norm = TwoSlopeNorm(vmin=average_interactions.min(), vcenter=0, vmax=average_interactions.max())
    plt.imshow(average_interactions, cmap='coolwarm', norm=norm, interpolation='nearest')
    plt.colorbar()  # カラーバーを追加
    plt.title('Heatmap of the Matrix')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    # ヒートマップをファイルに保存
    plt.savefig('ctx_interactions_heatmap.png', format='png')  # PNG形式で保存
    # リソースのクリーンアップ
    plt.close()

    # # 各入力画像での各トークンのSHAP値を格納するリスト
    # shap_values_list = []
    # for i, PIL_image in enumerate(PIL_image_list):
    #     # print(f"{i+1}個目の入力画像に対する各トークンのSHAP値を計算")
    #     shap_values = compute_ctx_shap_values(ctx_background_datas, ctx_model_predict, prompt_embedding, PIL_image)
    #     shap_values_list.append(shap_values) # len = num_input_images
    #     # print(shap_values[:12])
        
    # ave_shap_values = 0
    # shap_values_list_array = np.array(shap_values_list) # [3, 77]
    # # print(shap_values_list_array[:, :12])
    # ave_shap_values = shap_values_list_array.sum(axis=0, keepdims=True)
    # ave_shap_values /= len(shap_values_list) # [1, 77]
    # # print(ave_shap_values[0, :12])

    # shap_values_list3.append(ave_shap_values[0])

    # # 各トークンのSHAP値（平均）を表示
    # # print("\n学習されたプロンプトの各トークンのSHAP値（平均）")
    # # for i, ave_shap_value in enumerate(ave_shap_values[0]):
    # #     if  i < prompt_length:
    # #         if i != 0 and i != prompt_length - 1:
    # #             print(f"trained Token{i} average SHAP Value: {ave_shap_value:.5f}")

    # print("学習されたプロンプトの各トークンのSHAP値（各画像に対する平均）")
    # for i, ave_shap_value in enumerate(ave_shap_values[0]):
    #     if(shap_values[i] != 0 and i < ctx_length + 1):
    #         print(f"学習されたトークン{i} SHAP値: {ave_shap_value:.3f}")
    #     elif(shap_values[i] != 0):
    #         token_id = text_input[0, i].item()
    #         token = tokenizer.decode([token_id])
    #         print(f"{token:<15} SHAP値: {ave_shap_value:.3f}")
    # endregion


# region -------------各正解クラス、各画像に対して平均したSHAP値-------------
ave_shap_values2 = 0
shap_values_list_array2 = np.array(shap_values_list2) # [8, 77]
ave_shap_values2 = shap_values_list_array2.sum(axis=0, keepdims=True)
ave_shap_values2 /= len(shap_values_list2) # [1, 77]

init_ctx = args.init_ctx
text_input = clip.tokenize(f"{init_ctx} class.").to("cuda") # [1, 77]

print("初期値のプロンプトの各トークンのSHAP値（平均）")
for i, ave_shap_value2 in enumerate(ave_shap_values2[0]):
    if(ave_shap_value2 != 0):
        token_id = text_input[0, i].item()
        token = tokenizer.decode([token_id])
        print(f"{token:<15} SHAP値: {ave_shap_value2:.2f}")

ave_shap_values3 = 0
shap_values_list_array3 = np.array(shap_values_list3) # [8, 77]
ave_shap_values3 = shap_values_list_array3.sum(axis=0, keepdims=True)
ave_shap_values3 /= len(shap_values_list3) # [1, 77]

print("学習されたプロンプトの各トークンのSHAP値（平均）")
for i, ave_shap_value3 in enumerate(ave_shap_values3[0]):
    if(ave_shap_value3 != 0 and i < ctx_length+1):
        print(f"学習されたトークン{i} SHAP値: {ave_shap_value3:.3f}")
    elif(ave_shap_value3 != 0):
        token_id = text_input[0, i].item()
        token = tokenizer.decode([token_id])
        print(f"{token:<15} SHAP値: {ave_shap_value3:.3f}")
# endregion
