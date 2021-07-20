import numpy as np
from datetime import datetime
from layers import *
from collections import OrderedDict
from numerical_gradient import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # 가중치 초기화
        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()  # python의 모듈에서 불러온 메소드
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy
    
    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads



# Usage
train_data = np.loadtxt("C:/Users/csp/Desktop/MyGitHub/work2/neowizard/MachineLearning/mnist_train.csv", delimiter=',', dtype=np.float32)
test_data = np.loadtxt("C:/Users/csp/Desktop/MyGitHub/work2/neowizard/MachineLearning/mnist_test.csv", delimiter=',', dtype=np.float32)

input_size = train_data.shape[1]-1   # 784
hidden_size = 50
output_size = 10

input_train_data = train_data[:, 1:]
target_train_data = train_data[:, 0]
input_test_data = test_data[:, 1:]
target_test_data = test_data[:, 0]

input_train_list = []
input_test_list = []
target_train_list = []
target_test_list = []

# normalize (one-hot encoding)
for step in range(60000):
    input_train_list.append(((input_train_data[step, :] / 255.0)*0.99) + 0.01)
    target_train_list.append(np.zeros(output_size))
    target_train_list[step][int(target_train_data[step])] = 1
    
for step in range(10000):
    input_test_list.append(((input_test_data[step, :] / 255.0) * 0.99) + 0.01)
    target_test_list.append(np.zeros(output_size))
    target_test_list[step][int(target_test_data[step])] = 1

iters_num = 10000 
train_size = input_train_data.shape[0]
batch_size = 100
learning_rate = 1e-1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size, hidden_size, output_size)

start_time = datetime.now()

# 학습 시작
for i in range(iters_num):
    # 미니 배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    np_input_train_list = np.array(input_train_list)
    np_target_train_list = np.array(target_train_list)
    input_batch = np_input_train_list[batch_mask]
    target_batch = np_target_train_list[batch_mask]

    # 기울기 계산
    grad = network.gradient(input_batch, target_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(input_batch, target_batch)
    train_loss_list.append(loss)

    # 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(np_input_train_list, np_target_train_list)
        test_acc = network.accuracy(np.array(input_test_list), np.array(target_test_list))
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

end_time = datetime.now()

# 학습을 위해 수행한 총 시간 출력
print("\nElapsed time = ", end_time - start_time)

