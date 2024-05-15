'''
・ターミナルコマンド
  pyhton my_interpret.py {a/b/c/prompt_learner/model.pth.tar} {topk}
・データセットに応じて書き換える文
  ctx_init = {"a initialized prompt for a dataset"} 229行目
  plt.savefig('./tsne/trained_ctx_vocab_init_ctx_tsne_{dataset_name}.png') 271行目
'''

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import spacy
from sklearn.cluster import KMeans
import random

from clip.simple_tokenizer import SimpleTokenizer
from clip import clip

from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist

import numpy as np
from sklearn.preprocessing import normalize
from numpy.random import RandomState

class CosineKMeans:
    def __init__(self, n_clusters=100, max_iter=300, tol=0.0001, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        # 正規化してコサイン類似度を距離として使用できるようにする
        X = normalize(X)

        # RandomState インスタンスを作成
        rng = RandomState(self.random_state)

        # 初期の中心をランダムに選択
        indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[indices]

        for _ in range(self.max_iter):
            # 各点と中心とのコサイン類似度を計算
            similarities = np.dot(X, self.centroids.T)
            
            # 各点に最も類似する中心を見つける
            self.labels_ = np.argmax(similarities, axis=1)

            # 新しい中心を計算
            new_centroids = np.array([X[self.labels_ == k].mean(axis=0) for k in range(self.n_clusters)])
            new_centroids = normalize(new_centroids)  # 新しい中心も正規化

            # 収束チェック
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break

            self.centroids = new_centroids

        return self

    def predict(self, X):
        X = normalize(X)  # 入力データの正規化
        similarities = np.dot(X, self.centroids.T)
        return np.argmax(similarities, axis=1)

# CPUにclipモデルをロード
def load_clip_to_cpu(backbone_name="RN50"):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model

# GPUにclipモデルをロード
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

# region コマンドライン引数の設定
parser = argparse.ArgumentParser()
parser.add_argument("fpath", type=str, help="Path to the learned prompt")
parser.add_argument("topk", type=int, help="Select top-k similar words")
args = parser.parse_args()
fpath = args.fpath
topk = args.topk
assert os.path.exists(fpath) # ファイルの存在確認
print(f"Return the top-{topk} matched words")
# endregion

tokenizer = SimpleTokenizer()          # BPEトーカナイザー
clip_model = load_clip_to_gpu()        # 事前学習したCLIPをGPUにロード
embedding = clip_model.token_embedding # CLIPの事前学習でトレーニングしたトークン埋め込み層

# -------自然言語化-------
# 自然言語化用の辞書をロードし、トークン埋め込みベクトルに変換
with open('vocab_words.txt', 'r', encoding='utf-8') as file:     # .txtファイルから辞書を読み込む
    vocab_words = [line.strip() for line in file]
token_ids = [tokenizer.encode(word)[0] for word in vocab_words] # BPEトーカナイザーでID化
token_ids_tensor = torch.tensor(token_ids)
token_embedding = embedding(token_ids_tensor)                   # 辞書の各語のトークン埋め込みベクトルを取得
token_embedding = token_embedding.to("cuda")

# 学習prompt（ctx）の各トークンの埋め込みベクトルを取得
prompt_learner = torch.load(fpath, map_location="cuda")["state_dict"]
ctx = prompt_learner["ctx"]
ctx = ctx.float() # Size=([16, 512])

top_similar_words = [] # ctxトークン1に類似した上位topkの単語
# ctxの各トークンに類似した上位topkの単語を出力
for i, ctx_vec in enumerate(ctx): # コサイン類似度
    # 各ctxトークンと各辞書語のコサイン類似度を計算
    similarities = F.cosine_similarity(ctx_vec.unsqueeze(0), token_embedding, dim=-1)
    # 上位topkの類似度とそのインデックス
    top_similarities, top_indices = torch.topk(similarities, topk)
    print(f"学習されたトークン[V{i+1}]に類似した上位{topk}つの単語:")
    for j in range(topk):
        similar_word = vocab_words[top_indices[j]]
        similarity = top_similarities[j].item()
        print(f"    {j+1}: {similar_word} (類似度: {round(similarity, 2)})")
        # print(f"    {j+1}: {similar_word}")
        if i==0:
            top_similar_words.append(similar_word) # リストに保存
    print("-----------------------------------------------------")

top_near_words = [] # ctxトークン1に類似した上位topkの単語
# ctxの各トークンに類似した上位topkの単語を出力
for i, ctx_vec in enumerate(ctx): # ユークリッド距離
    euclidean_distance = torch.cdist(ctx_vec.unsqueeze(0), token_embedding)
    top_euclidean_distance, top_indices = torch.topk(euclidean_distance, topk, largest=False)
    print(f"ctxトークン {i+1} に類似した上位{topk}つの単語:")
    for j in range(topk):
        index = top_indices[0, j].item()
        near_word = vocab_words[index]
        nearity = top_euclidean_distance[0, j].item()
        # print(f"    {j+1}: {near_word} (類似度: {round(nearity, 2)})")
        print(f"    {j+1}: {near_word}")
        if i == 0:
            top_near_words.append(near_word) # リストに保存
    print("-----------------------------------------------------")

'''
# region -------cluster-------
# K-meansでクラスタリング
n_clusters = 30  # クラスタ数
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(token_embedding.detach().cpu().numpy())
labels = kmeans.labels_ # 各単語が属するクラスタのラベルを取得 len = len(vocab_words)
# クラスタごとに単語を表示
for cluster_id in range(n_clusters):
    print(f"Cluster {cluster_id}:")
    words_in_cluster = [vocab_words[i] for i in range(len(vocab_words)) if labels[i] == cluster_id]
    print(words_in_cluster[:10])

# # 正規化後にK-meansでクラスタリング
# n_clusters = 50
# cos_kmeans = KMeans(n_clusters=n_clusters, random_state=0)
# cos_kmeans.fit(normalize(token_embedding.detach().cpu().numpy()))
# cos_labels = kmeans.labels_
# # クラスタごとに単語を表示
# for cluster_id in range(n_clusters):
#     print(f"Cluster {cluster_id}:")
#     words_in_cluster = [vocab_words[i] for i in range(len(vocab_words)) if cos_labels[i] == cluster_id]
#     print(words_in_cluster[:10])

# コサイン類似度でクラスタリング
n_clusters = 30
cos_kmeans = CosineKMeans(n_clusters, random_state=0)
cos_kmeans.fit(token_embedding.cpu().detach().numpy())
cos_labels = cos_kmeans.labels_
# クラスタごとに単語を表示
for cluster_id in range(n_clusters):
    print(f"Cluster {cluster_id}:")
    words_in_cluster = [vocab_words[i] for i in range(len(vocab_words)) if cos_labels[i] == cluster_id]
    print(words_in_cluster[:10])

# 類似したtopkの単語の属するクラスタを表示
text = "type tests test sums today days minutes dates postcard beginners"
# テキストから単語を抽出
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])   # spaCyの埋め込み層
doc = nlp(text)
text_words = [token.text.lower() for token in doc if token.is_alpha]  # アルファベットのみの単語を小文字で抽出
# クラスタごとの単語リストを作成
clusters_words = {i: [] for i in range(n_clusters)}
for word, label in zip(vocab_words, cos_labels):
    clusters_words[label].append(word)
# テキスト内の各単語がどのクラスタに含まれるかをチェック
for i, word in enumerate(text_words):
    # if i % 3 == 0 and i != 0:
    #     print("-------------------------")
    found_cluster = False
    for cluster_id, words in clusters_words.items():
        if word in words:
            print(f"'{word}' is in Cluster: {cluster_id}")
            found_cluster = True
            break
    if not found_cluster:
        print(f"'{word}' is not included in any cluster.")
# endregion
'''

# -------T-SNE-------
# 初期promptのトークン埋め込みベクトルを取得
# ctx_init = "a flower photo of a" # oxford_flowersで学習する際の初期プロンプト
# ctx_init = "a color image of a"   # caltech101で学習する際の初期プロンプト
ctx_init = "the animal pet photo of a"   # caltech101で学習する際の初期プロンプト
n_ctx_init = len(ctx_init.split(" "))
ctx_init_id = clip.tokenize(ctx_init)
ctx_init_embedding = embedding(ctx_init_id[0, 1: n_ctx_init+1:])

# region ctxと辞書語を可視化
ctx_np = ctx.detach().cpu().numpy()
token_embedding_np = token_embedding.detach().cpu().numpy()
combined_data = np.concatenate([token_embedding_np, ctx_np], axis=0) # データの結合
tsne = TSNE(n_components=2, perplexity=10, random_state=0) # データを2次元に削減するtsneを定義
combined_data_2d = tsne.fit_transform(combined_data) # tsne実行
plt.figure(figsize=(30, 18)) # 図のサイズ
plt.scatter(combined_data_2d[:len(token_embedding_np), 0], combined_data_2d[:len(token_embedding_np), 1], label='vocabrary word') # 散布図にプロット
plt.scatter(combined_data_2d[len(token_embedding_np):, 0], combined_data_2d[len(token_embedding_np):, 1], label='trained context token')
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE Visualization of Two High-Dimensional Datasets')
plt.legend()
plt.rcParams.update({'font.size': 15})
plt.show()
plt.savefig('./tsne/trained_ctx_vocab_tsne.png')
# endregion

# region ctxと辞書語、init_ctxを可視化
ctx_init_embedding_np = ctx_init_embedding.detach().cpu().numpy()
combined_data_with_init = np.concatenate([combined_data, ctx_init_embedding_np], axis=0)
combined_data_with_init_2d = tsne.fit_transform(combined_data_with_init)
plt.figure(figsize=(30, 18))
plt.scatter(combined_data_with_init_2d[:len(token_embedding_np), 0], combined_data_with_init_2d[:len(token_embedding_np), 1], label='Vocabulary word')
plt.scatter(combined_data_with_init_2d[len(token_embedding_np):-len(ctx_init_embedding_np), 0], combined_data_with_init_2d[len(token_embedding_np):-len(ctx_init_embedding_np), 1], label='Trained token', s=75)
plt.scatter(combined_data_with_init_2d[-len(ctx_init_embedding_np):, 0], combined_data_with_init_2d[-len(ctx_init_embedding_np):, 1], label='Initial token', color='red', s=75)
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.xlabel('t-SNE Feature 1', fontsize=20)
plt.ylabel('t-SNE Feature 2', fontsize=20)
plt.title('The result of t-SNE: Flowers102', fontsize=30)
plt.legend(fontsize=30)
plt.show()
# plt.savefig('./tsne/trained_ctx_vocab_init_ctx_tsne_oxford_flowers.png')
# plt.savefig('./tsne/trained_ctx_vocab_init_ctx_tsne_oxford_flowers_random.png')
# plt.savefig('./tsne/trained_ctx_vocab_init_ctx_tsne_caltech101.png')
# endregion
'''
# region クラスタごとに辞書語を可視化
# # t-SNEのインスタンスを作成
# tsne = TSNE(n_components=2, perplexity=10, random_state=0)
# # プロットの設定
# plt.figure(figsize=(30, 18))
# # 各クラスタごとにプロット
# for cluster_id in range(n_clusters):
#     # クラスタに属するトークンのインデックスを取得
#     indices = np.where(labels == cluster_id)[0]
#     # クラスタ内の埋め込みベクトルを選択
#     cluster_embeddings = token_embedding_np[indices]
#     # t-SNEで2次元に削減
#     cluster_embeddings_2d = tsne.fit_transform(cluster_embeddings)
#     # 散布図にプロット（ここではランダムに色を選択）
#     plt.scatter(cluster_embeddings_2d[:, 0], cluster_embeddings_2d[:, 1], label=f'Cluster {cluster_id}')
# # 軸ラベルとタイトル
# plt.xlim(-100, 100)
# plt.ylim(-100, 100)
# plt.xlabel('t-SNE Feature 1')
# plt.ylabel('t-SNE Feature 2')
# plt.title('t-SNE Visualization of Token Embeddings by Cluster')
# plt.legend()
# plt.rcParams.update({'font.size': 20})
# plt.show()
# plt.savefig('./tsne/cluster_tsne.png')
# endregion
'''
# region 辞書語を可視化
tsne = TSNE(n_components=2, perplexity=10, random_state=0)
data_2d = tsne.fit_transform(token_embedding_np)
plt.figure(figsize=(30, 18))
plt.scatter(data_2d[:, 0], data_2d[:, 1])
plt.xlim(-100, 100)
plt.ylim(-100, 100)
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.title('t-SNE Visualization of High-Dimensional Data')
plt.show()
plt.savefig('./tsne/vocab_tsne.png')
# endregion

# region ctx[0]とctx[0]に類似した単語、辞書語を可視化
# t-SNEのインスタンスを作成
tsne = TSNE(n_components=2, perplexity=9, random_state=0)
# ctxトークン1の埋め込みベクトル
ctx_token_1_embedding = ctx[0].unsqueeze(0).detach().cpu().numpy()
ctx_token_1_embeddings = np.tile(ctx_token_1_embedding, (10, 1)) # 10つに複製
# ctxトークン1に類似した上位topkの単語の埋め込みベクトル
top_words = top_similar_words
top_word_ids = [tokenizer.encode(word)[0] for word in top_words]
top_word_tensors = torch.tensor(top_word_ids)
top_word_embeddings = embedding(top_word_tensors).detach().cpu().numpy()
print(top_words)
# 全辞書語の埋め込みベクトル
vocab_embeddings = token_embedding.detach().cpu().numpy()
# 類似した上位3つの単語とctxトークン1の埋め込みベクトルを結合
combined_embeddings = np.vstack([ctx_token_1_embeddings, top_word_embeddings, vocab_embeddings])
# t-SNEで2次元に削減-4
combined_embeddings_2d = tsne.fit_transform(combined_embeddings)
n_vocab = len(vocab_embeddings)
plt.figure(figsize=(30, 18))
plt.scatter(combined_embeddings_2d[-n_vocab:, 0], combined_embeddings_2d[-n_vocab:, 1], color='gray', label='Vocabulary word')
plt.scatter(combined_embeddings_2d[:10, 0], combined_embeddings_2d[:10, 1], color='blue', label='ctx Token 1')
plt.scatter(combined_embeddings_2d[10:20, 0], combined_embeddings_2d[10:20, 1], color='red', label='Topk Similar Word')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE Visualization of ctx Token 1 (Copies), Top Similar Words, and Vocabulary')
plt.legend()
plt.show()
plt.savefig('./tsne/aaa.png')
# endregion

# -------others-------
# init_ctxの各トークンに類似した上位topの単語を出力
for i, ctx_init_vec in enumerate(ctx_init_embedding.to("cuda")):
    similarities = F.cosine_similarity(ctx_init_vec.unsqueeze(0), token_embedding, dim=-1)
    top_similarities, top_indices = torch.topk(similarities, topk)  # 類似度の上位topkつとそのインデックス
    print(f"ctx_initトークン {i+1} に最も類似した上位{topk}つの単語:")
    for j in range(topk):
        similar_word = vocab_words[top_indices[j]]
        similarity = top_similarities[j].item()
        print(f"    {j+1}: {similar_word} (類似度: {round(similarity, 2)})")
    print("-----------------------------------------------------")

'''
# region 2つの単語の類似度を計算
word1 = "test"
word2 = "tests"
# トークンIDの取得
token_id_word1 = tokenizer.encode(word1)[0]
token_id_word2 = tokenizer.encode(word2)[0]
# トークンIDからテンソルを作成
token_tensor_word1 = torch.tensor([token_id_word1])
token_tensor_word2 = torch.tensor([token_id_word2])
# 埋め込みベクトルの取得
embedding_word1 = embedding(token_tensor_word1)
embedding_word2 = embedding(token_tensor_word2)
# GPUが利用可能な場合は、データをGPUに移動
if torch.cuda.is_available():
    embedding_word1 = embedding_word1.to("cuda")
    embedding_word2 = embedding_word2.to("cuda")
# 類似度の計算
similarity = F.cosine_similarity(embedding_word1, embedding_word2, dim=-1)
print(f"{word1} と {word2} の類似度: {round(similarity.item(), 3)}")
# endregion
'''
# region クエリ単語とctxの各トークンの類似度を計算
query_word = "color"
# クエリ単語のトークンIDと埋め込みベクトルを取得
query_token_id = tokenizer.encode(query_word)[0]
print(query_token_id)
query_token_tensor = torch.tensor([query_token_id])
query_embedding = embedding(query_token_tensor)
if torch.cuda.is_available():
    query_embedding = query_embedding.to("cuda")
# 類似度を計算
for i, ctx_vec in enumerate(ctx):
    similarity = F.cosine_similarity(ctx_vec.unsqueeze(0), query_embedding, dim=-1)
    print(f"ctxトークン {i+1} と '{query_word}' の類似度: {round(similarity.item(), 3)}")
# endregion
'''
# region ctx[?]の最も類似した単語とtop_similar_wordsのコサイン類似度を計算
query_word = "baggage"
query_token_id = tokenizer.encode(query_word)[0]
print(query_token_id)
query_token_tensor = torch.tensor([query_token_id])
query_embedding = embedding(query_token_tensor).to("cuda")
# top_similar_wordsの各単語の埋め込みベクトルを取得
top_word_embeddings = []
for word in top_similar_words:
    token_id = tokenizer.encode(word)[0]
    token_tensor = torch.tensor([token_id])
    word_embedding = embedding(token_tensor).to("cuda")
    top_word_embeddings.append(word_embedding.squeeze(0))
# 類似度計算
for i, word_embedding in enumerate(top_word_embeddings):
    similarity = F.cosine_similarity(query_embedding, word_embedding.unsqueeze(0), dim=-1)
    print(f"'type'と'{top_similar_words[i]}'の類似度: {round(similarity.item(), 3)}")
# endregion
'''
