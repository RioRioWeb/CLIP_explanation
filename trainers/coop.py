import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# tokenizerインスタンス
_tokenizer = _Tokenizer()

# 設定情報に基づいたCLIPをCPUに読み込む関数
def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]       # 指定されたバックボーンのurlを辞書から格納
    model_path = clip._download(url)        # urlからモデルのファイルをローカルに保存し、そのローカルパスを格納

    # モデルの状態辞書をロード
    try:
        # torch.jit.load()...TorchScriptファイル（JITアーカイブ）ロードする。この場合モデルはTorchScript形式で保存されている。
        # .eval()...モデルを評価モードに設定し、トレーニング特有の挙動（例：ドロップアウト）を無効化
        # map_location...データどのデバイスにロードするか。cudaならGPU。
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        # state_dict...モデルのパラメータ（重みやバイアスなど）を格納するために使用される辞書
        state_dict = torch.load(model_path, map_location="cpu")

    # 与えられた状態辞書からCLIPモデルを作成する
    model = clip.build_model(state_dict or model.state_dict())

    return model

# nn.Moduleを継承し、TextEncoderモデルを構築する
# nn.Module...ニューラルネットワークライブラリ（torch.nn）のクラスで、ニューラルネットワークモデルの構築に用いられる親クラス。
class TextEncoder(nn.Module):
    # クラスの初期化時に既存のclipモデルからいくつかのコンポーネントを取り込む
    # self...インスタンス自身を参照するためのもの
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final                         # 最終的な正規化層
        self.text_projection = clip_model.text_projection           # テキスト射影...テキストデータの特徴を画像データの特徴空間に合わせる
        self.dtype = clip_model.dtype                               # データ型（dtype)はテンソル内の要素の型を指す。例えば、torch.float32, torch.float64, torch.int32。

    # 前向き伝播（Forward Pass）を上書きして、入力データがモデルを通過する際の計算手順を指定。
    def forward(self, prompts, tokenized_prompts):              # prompts(143行): 特定のクラス名に対応する学習テキストの埋め込みベクトル（✖︎クラス数）
                                                                # tokenized_prompts(114, 117行目): 特定のクラス名に対応するctx_initのトークンベクトル（✖︎クラス数）
        x = prompts + self.positional_embedding.type(self.dtype)
        # permute()...テンソルの次元を入れ替える。トランスフォーマーモデルが入力の次元順序としてLNDを期待するため必要。
        x = x.permute(1, 0, 2)                                  # NLD -> LND バッチサイズ（N）、シーケンス（コンテキスト）長（L）、特徴次元（D）
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                                  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the EOT (End Of Text) embedding (EOT_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
# コンテキストベクトルを学習し、生成するためのクラス
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)                          # クラス数
        n_ctx = cfg.TRAINER.COOP.N_CTX                   # コンテキストトークンの数
        ctx_init = cfg.TRAINER.COOP.CTX_INIT             # コンテキストの初期化方法
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]    # コンテキストトークンベクトルの次元数（＝CLIPの最終出力の特徴ベクトルの次元数）
        clip_imsize = clip_model.visual.input_resolution # 解像度
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        # ランダム初期値のctx
        global random_init_ctx

        # use given words to initialize context vectors
        # ctx_initを設定した場合、n_ctxは16以下。例）a photo of aなら、4つのトークンから成るctxを最適化
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)                               # トークン化
            with torch.no_grad():                                          # 勾配計算の無効化。既存のモデルを使用してトークン埋め込みベクトルを取得するためのものなので、勾配計算は不要。
                embedding = clip_model.token_embedding(prompt).type(dtype) # トークン埋め込み
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]                   # 指定されたトークン数までを抽出
            prompt_prefix = ctx_init

        # random initialization
        # ctx_initを設定しない場合、n_ctx個のctxトークンがランダムに生成される
        else:
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            print(f'init_ctx: {ctx_vectors}')       ###
            print(f'init_ctx: {ctx_vectors.shape}') ###
            random_init_ctx = ctx_vectors           ###
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # nn.Parameterとして宣言されたテンソルはモデルのパラメータとして登録され、学習の際にオプティマイザに更新される
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        # 特定のクラス名に対応するテキストプロンプトを作成
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # 作成したプロンプトから（IDベクトル→）トークン埋め込みベクトルを取得
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # register_buffer()...モデルのパラメータではないが、モデルの状態の一部として保存されるべきデータを登録
        self.register_buffer("token_prefix", embedding[:, :1, :])           # SOSトークンを取得し、登録
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLSとEOSトークンを取得し、登録

        # depcriptor = "flower"
        # descriptor_id = _tokenizer.encode(depcriptor)
        # descriptor_embedding = clip_model.token_embedding(torch.tensor(descriptor_id)).type(dtype).unsqueeze(0).expand(16, -1, -1)
        # self.register_buffer("token_descriptor", descriptor_embedding)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    # 学習するコンテキストベクトルを使用して、特定のクラス名に対応するプロンプトを生成。コンテキストベクトルは、プレフィックスとサフィックスとともに結合され、クラス名ごとの完全なプロンプトを形成。
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        # descriptor = self.token_descriptor ###

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    # descriptor, ###
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

# PromptLearnerとCLIPモデルの両方を組み合わせたCLIPモデルを形成
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model) ###
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts   ###
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)                      ###
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    # text promptとimageの類似度を計算
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()                                  ###
        tokenized_prompts = self.tokenized_prompts                       ###
        text_features = self.text_encoder(prompts, tokenized_prompts)    ###

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits

# 学習関連のクラスや関数を管理するためのレジストリにCoOpクラスを登録
@TRAINER_REGISTRY.register()
# 学習を制御するクラス
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    # fp16, fp32, ampから数値精度を設定
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model) ###

        # text/image encoderのパラメータを固定
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        #prompt_learnerモジュールに重みをロード
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        # NOTE: only give prompt_learner to the optimizer
        self.model.to(self.device)                                                               # モデルをデバイス（CPU, GPU）に移動
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)                       # prompt_learnerモジュール専用の最適化器を構築
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)                                   # 学習率スケジューラを構築
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched) # promptモジュールをトレーナークラスに登録

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None                   # 自動混合精度（AMP）スケーラの設定

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)