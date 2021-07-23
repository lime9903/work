import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)


x = np.random.randn(1000, 100)  # 1000개의 데이터 (1000*100)
node_num = 100              # 각 층의 노드 수
hidden_layer_size = 5       # 은닉층 5개
activations = {}            # 활성화값(활성화 함수의 출력 데이터)을 저장할 변수

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    
    w = np.random.randn(node_num, node_num) * np.sqrt(2 / node_num)

    a = np.dot(x, w)
    z = relu(a)
    activations[i] = z

# 히스토그램 그리기
for i, res in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])    # 1번째 그래프에만 y축 나타내기
    plt.ylim(0, 7000)
    plt.hist(res.flatten(), 30, range=(0, 1))
plt.show()