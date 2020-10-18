import numpy as np
import random

class Node:

    def __init__(self):
        pass

    def compute(self, *args):
        raise NotImplemented

    def grad_fn(self, *args):
        raise NotImplemented

    def tensor_trans(self, x):
        if type(x) != Tensor:
            x = Tensor(x)
        return x

    def get_requires_grad_status(self, *args):
        requires_grad = False
        for t in args:
            if type(t) == Tensor:
                if t.requires_grad == True:
                    requires_grad = True
                    break
        return requires_grad


class Add(Node):

    def __init__(self):
        self.result = None
        self.x = None
        self.y = None

    def compute(self, x, y):
        self.x = self.tensor_trans(x)
        self.y = self.tensor_trans(y)
        requires_grad = self.get_requires_grad_status(self.x, self.y)
        self.result = Tensor(self.x._value + self.y._value, requires_grad=requires_grad)
        self.result.grad_fn = self.grad_fn
        return self.result

    def grad_fn(self, parent_grad):
        if self.x.requires_grad:
            if self.x._grad is None:
                self.x._grad = Tensor(0)
            self.x._grad += parent_grad
        if self.y.requires_grad:
            if self.y._grad is None:
                self.y._grad = Tensor(0)
            self.y._grad += parent_grad
        return self.x, self.y


class Sub(Node):

    def __init__(self):
        self.result = None
        self.x = None
        self.y = None

    def compute(self, x, y):
        self.x = self.tensor_trans(x)
        self.y = self.tensor_trans(y)
        requires_grad = self.get_requires_grad_status(self.x, self.y)
        self.result = Tensor(self.x._value - self.y._value, requires_grad=requires_grad)
        self.result.grad_fn = self.grad_fn
        return self.result

    def grad_fn(self, parent_grad):
        if self.x.requires_grad:
            if self.x._grad is None:
                self.x._grad = Tensor(0)
            self.x._grad += parent_grad
        if self.y.requires_grad:
            if self.y._grad is None:
                self.y._grad = Tensor(0)
            self.y._grad += -parent_grad
        return self.x, self.y


class Mul(Node):

    def __init__(self):
        self.result = None
        self.x = None
        self.y = None

    def compute(self, x, y):
        self.x = self.tensor_trans(x)
        self.y = self.tensor_trans(y)
        requires_grad = self.get_requires_grad_status(self.x, self.y)
        self.result = Tensor(self.x._value * self.y._value, requires_grad=requires_grad)
        self.result.grad_fn = self.grad_fn
        return self.result

    def grad_fn(self, parent_grad):
        if self.x.requires_grad:
            if self.x._grad is None:
                self.x._grad = Tensor(0)
            self.x._grad += parent_grad * self.y._value
        if self.y.requires_grad:
            if self.y._grad is None:
                self.y._grad = Tensor(0)
            self.y._grad += parent_grad * self.x._value
        return self.x, self.y


class Div(Node):

    def __init__(self):
        self.result = None
        self.x = None
        self.y = None

    def compute(self, x, y):
        self.x = self.tensor_trans(x)
        self.y = self.tensor_trans(y)
        requires_grad = self.get_requires_grad_status(self.x, self.y)
        self.result = Tensor(self.x._value / self.y._value, requires_grad=requires_grad)
        self.result.grad_fn = self.grad_fn
        return self.result

    def grad_fn(self, parent_grad):
        if self.x.requires_grad:
            if self.x._grad is None:
                self.x._grad = Tensor(0)
            self.x._grad += parent_grad * (1 / self.y._value)
        if self.y.requires_grad:
            if self.y._grad is None:
                self.y._grad = Tensor(0)
            self.y._grad += parent_grad * ((-self.x._value) * (self.y._value ** (-2)))
        return self.x, self.y


class Power(Node):

    def __init__(self):
        self.result = None
        self.x = None
        self.y = None

    def compute(self, x, y):
        self.x = self.tensor_trans(x)
        self.y = self.tensor_trans(y)
        requires_grad = self.get_requires_grad_status(self.x, self.y)
        self.result = Tensor(self.x._value ** self.y._value, requires_grad=requires_grad)
        self.result.grad_fn = self.grad_fn
        return self.result

    def grad_fn(self, parent_grad):
        if self.x.requires_grad:
            if self.x._grad is None:
                self.x._grad = Tensor(0)
            self.x._grad += parent_grad * (self.y._value * (self.x._value ** (self.y._value - 1)))
        if self.y.requires_grad:
            if self.y._grad is None:
                self.y._grad = Tensor(0)
            self.y._grad += parent_grad * (self.x._value ** self.y._value * (math.log(self.x._value + e - 7)))
        return self.x, self.y

# ReLU激活函数
class ReLU(Node):

    def __init__(self):
        self.result = None
        self.x = None

    def compute(self, x):
        self.x = self.tensor_trans(x)
        requires_grad = self.get_requires_grad_status(self.x)
        if self.x._value > 0:
            self.result = Tensor(self.x._value, requires_grad=requires_grad)
        else:
            self.result = Tensor(0, requires_grad=requires_grad)

        self.result.grad_fn = self.grad_fn
        return self.result

    def grad_fn(self, parent_grad):
        if self.x.requires_grad:
            if self.x._grad is None:
                self.x._grad = Tensor(0)

            if self.x._value > 0:
                self.x._grad += parent_grad
            else:
                self.x._grad += Tensor(0)
        return self.x


# 定义leakReLU激活函数
class LeakyReLU(Node):

    def __init__(self):
        self.result = None
        self.x = None

    def compute(self, x):
        self.x = self.tensor_trans(x)
        requires_grad = self.get_requires_grad_status(self.x)
        if self.x._value > 0:
            self.result = Tensor(self.x._value, requires_grad=requires_grad)
        else:
            self.result = Tensor(0.01 * self.x._value, requires_grad=requires_grad)

        self.result.grad_fn = self.grad_fn
        return self.result

    def grad_fn(self, parent_grad):
        if self.x.requires_grad:
            if self.x._grad is None:
                self.x._grad = Tensor(0)

            if self.x._value > 0:
                self.x._grad += parent_grad
            else:
                self.x._grad += parent_grad * 0.01
        return self.x

# logistic激活函数
class Logistic(Node):

    def __init__(self):
        self.result = None
        self.x = None

    def compute(self, x):
        self.x = self.tensor_trans(x)
        requires_grad = self.get_requires_grad_status(self.x)

        # 为了稳定数值
        if self.x._value > 15:
            self.result = Tensor(1e-8, requires_grad=requires_grad)
        elif self.x._value < -15:
            self.result = Tensor(0.9999997, requires_grad=requires_grad)
        else:
            self.result = Tensor(1 / (1 + math.e ** (- self.x._value)), requires_grad=requires_grad)

        self.result.grad_fn = self.grad_fn
        return self.result

    def grad_fn(self, parent_grad):
        if self.x.requires_grad:
            if self.x._grad is None:
                self.x._grad = Tensor(0)
            self.x._grad += parent_grad * self.result._value * (1 - self.result._value)
        return self.x

# 定义交叉熵类
class CrossEntropy(Node):

    def __init__(self):
        self.result = None
        self.inputs = None
        self.labels = None

    def compute(self, inputs, labels):
        self.inputs = [self.tensor_trans(i) for i in inputs]
        self.labels = [self.tensor_trans(i) for i in labels]
        requires_grad = self.get_requires_grad_status(*self.inputs)
        result = 0
        for i, l in zip(self.inputs, self.labels):
            result += -l._value * math.log(i._value + 1e-7)
        self.result = Tensor(result, requires_grad=requires_grad)
        self.result.grad_fn = self.grad_fn
        return self.result

    def grad_fn(self, parent_grad):
        for i, l in zip(self.inputs, self.labels):
            if i.requires_grad:
                if i._grad is None:
                    i._grad = Tensor(0)
                i._grad += parent_grad * (- l._value / (i._value))
        return self.inputs

# 定义softmax交叉熵，进行对多分类问题的性能评价
class SoftmaxCrossEntropy(Node):

    def __init__(self):
        self.result = None
        self.inputs = None
        self.softmax_input = None
        self.labels = None

    def compute(self, inputs, labels):
        self.inputs = [self.tensor_trans(i) for i in inputs]
        self.labels = [self.tensor_trans(i) for i in labels]
        requires_grad = self.get_requires_grad_status(*self.inputs)

        tmp = []
        for i in self.inputs:
            if i._value > 15:
                v = 15
            elif i._value < -15:
                v = -15
            else:
                v = i._value
            tmp.append(math.e ** v)
        total = 0
        for i in tmp:
            total += i
        self.softmax_input = [Tensor(i / total) for i in tmp]
        result = 0
        for i, l in zip(self.softmax_input, self.labels):
            result += -l._value * math.log(i._value + 1e-7)
        self.result = Tensor(result, requires_grad=requires_grad)
        self.result.grad_fn = self.grad_fn
        return self.result

    def grad_fn(self, parent_grad):
        for i, s, l in zip(self.inputs, self.softmax_input, self.labels):
            if i.requires_grad:
                if i._grad is None:
                    i._grad = Tensor(0)
                i._grad += parent_grad * (s._value - l._value)
        return self.inputs

# 重写运算符
class op:

    @staticmethod
    def add(x, y):
        return Add().compute(x, y)

    @staticmethod
    def sub(x, y):
        return Sub().compute(x, y)

    @staticmethod
    def mul(x, y):
        return Mul().compute(x, y)

    @staticmethod
    def div(x, y):
        return Div().compute(x, y)

    @staticmethod
    def pow(x, y):
        return Power().compute(x, y)

    @staticmethod
    def relu(x):
        return ReLU().compute(x)

    @staticmethod
    def leaky_relu(x):
        return LeakyReLU().compute(x)

    @staticmethod
    def logistic(x):
        return Logistic().compute(x)

    @staticmethod
    def cross_entropy(inputs, labels):
        return CrossEntropy().compute(inputs, labels)

    @staticmethod
    def softmax_cross_entropy(logits, labels):
        return SoftmaxCrossEntropy().compute(logits, labels)

# Tensor类
class Tensor:

    def __init__(self, value, requires_grad=False):

        if type(value) != Tensor:
            self._value = value
        else:
            self._value = value._value

        self.requires_grad = requires_grad
        self._grad = None
        self.grad_fn = None

    @property
    def grad(self):
        if self.requires_grad:
            return self._grad
        else:
            return None

    @grad.setter
    def grad(self, value):
        self._grad = value

    def zero_grad(self):
        self._grad = Tensor(0)

    def backward(self, parent_grad=None):
        if parent_grad is None:
            parent_grad = Tensor(1)

        if self.requires_grad is True and self.grad_fn is not None:
            tensors = self.grad_fn(parent_grad)
            if type(tensors) == Tensor:
                tensors = [tensors]
            for t in tensors:
                if t._grad is None:
                    p = Tensor(1)
                else:
                    p = t._grad.copy()
                t.backward(parent_grad=p)

    def copy(self):
        return Tensor(self._value)

    def __str__(self):
        return 'Tensor(' + str(self._value) + ')'

    def __repr__(self):
        return self.__str__()

    def __add__(self, y):
        return op.add(self, y)

    def __iadd__(self, y):
        return op.add(self, y)

    def __radd__(self, y):
        return op.add(y, self)

    def __pos__(self):
        return op.add(0, y)

    def __sub__(self, y):
        return op.sub(self, y)

    def __isub__(self, y):
        return op.sub(self, y)

    def __rsub__(self, y):
        return op.sub(y, self)

    def __neg__(self):
        return op.sub(0, self)

    def __mul__(self, y):
        return op.mul(self, y)

    def __imul__(self, y):
        return op.mul(self, y)

    def __rmul__(self, y):
        return op.mul(y, self)

    def __truediv__(self, y):
        return op.div(self, y)

    def __itruediv__(self, y):
        return op.div(self, y)

    def __rtruediv__(self, y):
        return op.div(y, self)

    def __pow__(self, y):
        return op.pow(self, y)

    def __ipow__(self, y):
        return op.pow(self, y)

    def __rpow__(self, y):
        return op.pow(y, self)

# 测试
if __name__ == "__main__":
    # 1. 前向传播

    # `requires_grad=True` 表明需要求对 `var_1` 的偏导数
    var_1 = Tensor(3, requires_grad=True)
    var_2 = Tensor(3, requires_grad=True)

    result = var_1 + var_2 ** 2  # 也可以写成 result = op.add(var_1, var_2)
    # result 对二者的偏导数计算公式
    print(result.grad_fn)

    # var_1, var_2 此时的偏导数均为 `None` (前向传播，还有没有生成导数)
    print(var_1.grad, var_2.grad)

    # 2. 反向传播

    # 执行反向传播
    result.backward()
    # var_1, var_2 此时的偏导数均为 `1`
    print(var_1.grad, var_2.grad)