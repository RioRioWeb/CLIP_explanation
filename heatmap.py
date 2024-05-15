import numpy as np
import matplotlib.pyplot as plt

# 二次元配列データ
data = np.array([
    [0.0, -0.0, -0.0, -0.1, -0.2, 0.2, 0.2],
    [-0.0, 0.0, -0.9, 0.4, -0.4, -1.5, 0.1],
    [-0.0, -0.9, 0.0, 0.1, -0.3, 0.0, 0.2],
    [-0.1, 0.4, 0.1, 0.0, -0.1, -0.1, 0.1],
    [-0.2, -0.4, -0.3, -0.1, 0.0, -0.7, 0.2],
    [0.2, -1.5, 0.0, -0.1, -0.7, 0.0, 0.1],
    [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.0]
])

# ヒートマップを生成
plt.imshow(data, cmap='coolwarm', interpolation='nearest')
plt.colorbar()  # カラーバーを追加
plt.title('Heatmap of the Matrix')
plt.xlabel('Column Index')
plt.ylabel('Row Index')

# ヒートマップをファイルに保存
plt.savefig('heatmap.png', format='png')  # PNG形式で保存

# プロットの表示は必要ない場合はコメントアウトする
# plt.show()

# リソースのクリーンアップ
plt.close()
