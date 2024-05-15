import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoTokenizer

# 仮のShapley Valueリスト
Shapley_Value_list = np.random.randn(10)

# カスタムカラーマップの作成
colors = [(0, 'blue'), (0.5, 'white'), (1, 'red')]  # 赤から青へのカラーマップ
cmap_name = 'shapley_value_cmap'
cm = LinearSegmentedColormap.from_list(cmap_name, colors)

# トークンのサンプルテキストとトークン化
text = "This is a sample text to tokenize."
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(text)

# トークンごとにShapley Valueに基づいて色を設定し、表示
plt.figure(figsize=(10, 1))
plt.xticks([])
plt.yticks([])
for idx, token in enumerate(tokens):
    color_value = (Shapley_Value_list[idx] - np.min(Shapley_Value_list)) / (np.max(Shapley_Value_list) - np.min(Shapley_Value_list))
    plt.text(idx, 0, token, color=cm(color_value), ha='center', fontsize=12)

# カラーバーの表示
plt.colorbar(plt.cm.ScalarMappable(cmap=cm), orientation='horizontal', label='Shapley Value')
# ヒートマップをファイルに保存
plt.savefig('shapley_value.png', format='png')  # PNG形式で保存
# リソースのクリーンアップ
plt.close()