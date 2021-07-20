import numpy as np
from datetime import datetime

# 수치 미분
def numerical_diff(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)
        
        x[idx] = tmp_val
        it.iternext()
    
    return grad

# sigmoid 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# softmax 함수
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

# cross-entropy
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    delta = 1e-7    
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+delta)) / batch_size

# 신경망 클래스
class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size):

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) / np.sqrt(input_size/2)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size/2)
        self.params['b2'] = np.zeros(output_size)

        print("Two Layer Network object is created!")

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        A1 = np.dot(x, W1) + b1
        Z1 = sigmoid(A1)
        A2 = np.dot(Z1, W2) + b2
        y = softmax(A2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def train(self, x, t):
        loss_function = lambda W : self.loss(x, t)

        # 기울기 update
        grads = {}
        grads['W1'] = numerical_diff(loss_function, self.params['W1'])
        grads['b1'] = numerical_diff(loss_function, self.params['b1'])
        grads['W2'] = numerical_diff(loss_function, self.params['W2'])
        grads['b2'] = numerical_diff(loss_function, self.params['b2'])

        return grads



# 데이터 불러오기
train_data = np.loadtxt("C:/Users/user/Desktop/work/Deep-Learning-from-Scratch/Chapter_4/mnist_train.csv", delimiter=',', dtype=np.float32)
test_data = np.loadtxt("C:/Users/user/Desktop/work/Deep-Learning-from-Scratch/Chapter_4/mnist_test.csv", delimiter=',', dtype=np.float32)

# layer size 정의
input_size = train_data.shape[1]-1   # 784
hidden_size = 50
output_size = 10

# train data와 test data를 input과 target으로 나누어주기
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

# 하이퍼 파라미터
iters_num = 10000  # 반복 횟수
train_size = input_train_data.shape[0]  # 60000
batch_size = 100   # 미니 배치 크기
learning_rate = 1e-1
epochs = 1

# 기록
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 신경망 생성
network = TwoLayerNetwork(input_size, hidden_size, output_size)

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
    grad = network.train(input_batch, target_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(input_batch, target_batch)
    train_loss_list.append(loss)

    # 정확도 계산
    if i % 10 == 0:
        train_acc = network.accuracy(np_input_train_list, np_target_train_list)
        test_acc = network.accuracy(np.array(input_test_list), np.array(target_test_list))
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

end_time = datetime.now()

# 학습을 위해 수행한 총 시간 출력
print("\nElapsed time = ", end_time - start_time)