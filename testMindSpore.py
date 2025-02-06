import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.nn as nn

# 打印MindSpore版本
print(mindspore.__version__)

# 创建一个张量
x = Tensor([1, 2, 3], mindspore.int32)

# 创建一个简单的神经网络层
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = ops.Add()

    def construct(self, x, y):
        return self.add(x, y)

# 实例化网络并进行前向传播
net = Net()
y = Tensor([4, 5, 6], mindspore.int32)
result = net(x, y)
print(result)