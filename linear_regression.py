import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

true_w = 2.0
true_b = 5.0

x = np.random.uniform(0, 10, 100)

noise = np.random.randn(100)
y = true_w * x + true_b + noise

print(f"✅ 第 1 部分：已生成 100 个样本")
print(f"   真实公式: y = {true_w} * x + {true_b} + 噪声")


def compute_loss(Xb, y, w):
    m = len(y)
    return (1/(2*m)) * np.sum((Xb @ w - y)**2)


class LinearRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.theta_0 = 0
        self.theta_1 = 0
        self.cost = []

    def train(self, x, y):
        self.theta_0 = 0
        self.theta_1 = 0
        self.cost = []

        x = np.asarray(x).reshape(-1, 1)
        m = x.shape[0]
        Xb = np.hstack([np.ones((m, 1)), x])
        w = np.zeros(2)

        for i in range(self.n_iter):
            idx = np.random.randint(0, m)
            xi = Xb[idx:idx+1]
            pred = xi @ w

            grad = xi.T @ (pred - y[idx])

            w = w - self.lr * grad.flatten()

            self.theta_0 = w[0]
            self.theta_1 = w[1]

            self.cost.append(compute_loss(Xb, y, w))

        return self

    def predict(self, x):
        x = np.asarray(x).reshape(-1)
        return self.theta_0 + self.theta_1 * x

    def print_result(self):
        print("=" * 50)
        print(f"📊 训练完成！（优化算法: SGD）")
        print(f"   学到的斜率 w (theta_1) = {self.theta_1:.4f}")
        print(f"   学到的截距 b (theta_0) = {self.theta_0:.4f}")
        print(f"   最终损失 = {self.cost[-1]:.6f}")
        print("-" * 50)
        print(f"🎯 真实值:  w = {true_w:.4f},  b = {true_b:.4f}")
        print(f"📏 误差:    Δw = {abs(true_w - self.theta_1):.4f}, "
              f"Δb = {abs(true_b - self.theta_0):.4f}")
        print("=" * 50)


model = LinearRegression(lr=0.01, n_iter=100)
model.train(x, y)
model.print_result()



plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', alpha=0.8, label='数据点')
plt.plot(x, model.predict(x), color='red', linewidth=2, label='学到的直线')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'拟合结果 (SGD): y = {model.theta_1:.2f}x + {model.theta_0:.2f}')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(len(model.cost)), model.cost, color='green')
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.title('损失随训练下降 (SGD 会有抖动)')
plt.grid(True)

plt.tight_layout()
plt.show()
