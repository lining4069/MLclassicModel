import numpy as np
import matplotlib.pyplot as plt

class SoftmaxModel():
    def __init__(self, train_txt_filename, test_txt_filename,n_class):
        self.train_txt_filename = train_txt_filename
        self.test_txt_filename = test_txt_filename
        self.n_class=n_class

    # 一、加载数据
    def load_dataset(self,file_path):
        data_arr = []
        label_arr = []
        fr = open(file_path)
        for line in fr.readlines():
            line_arr = line.strip().split()
            data_arr.append([1.0, float(line_arr[0]), float(line_arr[1])])
            label_arr.append(int(line_arr[2]))
        return data_arr, label_arr

    # 二、训练权重

    # 独热编码
    def one_hot(self,label_arr, n_samples, n_classes):
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), label_arr.T] = 1
        return one_hot

    # softmax函数
    def softmax(self,scores):
        # 计算总和
        sum_exp = np.sum(np.exp(scores), axis=1)
        softmax = np.exp(scores) / sum_exp
        return softmax

    # 梯度下降训练模型，获得参数矩阵weights以及每次迭代的损失loss
    def train(self,data_arr, label_arr, n_class, iters=1000, alpha=0.1, lam=0.01):
        n_samples, n_features = data_arr.shape
        n_classes = n_class
        # 随机初始化权重矩阵
        weights = np.random.rand(n_class, n_features)

        # 定义损失结果
        all_loss = []
        # 计算 one-hot 矩阵(独热编码)
        y_one_hot = self.one_hot(label_arr, n_samples, n_classes)
        for i in range(iters):
            scores = data_arr * weights.T
            # 计算 softmax 的值
            probs = self.softmax(scores)
            # 计算损失函数值,np.multiply对应位置相乘
            loss = - (1.0 / n_samples) * np.sum(np.multiply(y_one_hot, np.log(probs)))
            all_loss.append(loss)
            # 求解梯度
            dw = -(1.0 / n_samples) * ((y_one_hot - probs).T * data_arr)
            # 更新权重矩阵
            weights = weights - alpha * dw
        return weights, all_loss

    # 三、进行预测

    # 根据测试集矩阵test_dataset，以模型参数weights
    def predict(self,test_dataset, weights):
        scores = test_dataset * weights.T
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)

    # 测试函数 根据测试文件的filename以及训练出的模型的参数来进行测试，
    # 出根据参数weights构建的softmax分类模型（多分类逻辑斯蒂回归模型）的预测正确率以及迭代过程的loss-train_iter图像
    def test(self,filename, weights,all_loss):
        # 计算预测的准确率
        test_data_arr, test_label_arr = self.load_dataset(filename)
        test_data_arr = np.mat(test_data_arr)
        test_label_arr = np.mat(test_label_arr).T
        n_test_samples = test_data_arr.shape[0]
        y_predict = self.predict(test_data_arr, weights)
        accuray = np.sum(y_predict == test_label_arr) / n_test_samples
        print("In this softmax model the accuary for the test is {}".format(accuray))

        # 绘制损失函数
        fig = plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1000), all_loss)
        plt.title("loss - train iter")
        plt.xlabel("train iter")
        plt.ylabel("loss")
        plt.show()

    def run(self):
        data_arr, label_arr = self.load_dataset(self.train_txt_filename)
        data_arr = np.mat(data_arr)
        label_arr = np.mat(label_arr).T
        weights, all_loss = self.train(data_arr, label_arr, n_class=self.n_class)
        self.test(self.test_txt_filename, weights,all_loss)

if __name__ == "__main__":
    # save_train_txt_filename = "./data/train_dataset.txt"
    # save_test_txt_filename = "./data/test_dataset.txt"
    # gen_dataset(save_train_txt_filename, save_test_txt_filename)

    train_txt_filename="./data/train_dataset.txt"
    test_txt_filename="./data/test_dataset.txt"
    n_class=4
    model=SoftmaxModel(train_txt_filename=train_txt_filename,test_txt_filename=test_txt_filename,n_class=n_class)
    model.run()

