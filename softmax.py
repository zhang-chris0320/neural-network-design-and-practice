import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据
np.random.seed(42)
X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.1, random_state=42)
X = (X - X.mean(axis=0)) / X.std(axis=0)
Y = np.eye(3)[y]  # one-hot

idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
Y_train = Y[idx[:split]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

# 2. Softmax
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# 3. 训练
W = np.zeros((2, 3))
b = np.zeros(3)
lr, n_iter = 0.1, 1000

for i in range(n_iter):
    Y_hat = softmax(X_train @ W + b)
    W -= lr * X_train.T @ (Y_hat - Y_train) / len(X_train)
    b -= lr * (Y_hat - Y_train).mean(axis=0)

# 4. 预测
def predict(X):
    return np.argmax(softmax(X @ W + b), axis=1)

print(f"测试准确率: {np.mean(predict(X_test) == y_test)*100:.1f}%")

# 5. 可视化
fig, ax = plt.subplots(figsize=(7, 5))
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
Z = predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, levels=[-.5, .5, 1.5, 2.5], cmap='Pastel1')
for c in range(3):
    ax.scatter(X[y==c, 0], X[y==c, 1], label=f'类别{c}', alpha=0.8)
ax.set_title('Softmax 决策边界')
ax.legend(); ax.grid(True)
plt.savefig('softmax结果.png', dpi=120, bbox_inches='tight')
print("✅ 图片已保存")