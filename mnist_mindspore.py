import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn, Tensor, context
from mindspore.train.callback import LossMonitor
from mindspore.dataset import transforms
from mindspore.dataset.vision import transforms as vision_transforms
import matplotlib.pyplot as plt

# 1. 设置运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# 2. 数据预处理和加载
def create_dataset(batch_size=64):
    transform = [
        vision_transforms.Rescale(1.0 / 255.0, 0),   # 归一化到 [0,1]
        vision_transforms.Normalize(mean=(0.5,), std=(0.5,)),  # 标准化
        vision_transforms.HWC2CHW()  # 转换为 [C, H, W]
    ]

    # 加载 MNIST 数据集
    train_data = ds.MnistDataset(dataset_dir="D:/DeskTop/homework2/data/MNIST", usage='train', shuffle=True)
    test_data = ds.MnistDataset(dataset_dir="D:/DeskTop/homework2/data/MNIST", usage='test')


    train_data = train_data.map(operations=transform, input_columns="image")
    train_data = train_data.map(operations=transforms.TypeCast(ms.int32), input_columns="label")

    test_data = test_data.map(operations=transform, input_columns="image")
    test_data = test_data.map(operations=transforms.TypeCast(ms.int32), input_columns="label")


    train_data = train_data.batch(batch_size)
    test_data = test_data.batch(batch_size)

    return train_data, test_data

# 3. 定义 CNN 模型
class CNN(nn.Cell):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, pad_mode='pad', padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, pad_mode='pad', padding=1)
        self.fc1 = nn.Dense(32 * 7 * 7, 128)
        self.fc2 = nn.Dense(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 4. 模型训练
def train_model():
    # 加载数据
    train_data, test_data = create_dataset()

    # 定义网络和超参数
    network = CNN()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(network.trainable_params(), learning_rate=0.001)
    model = ms.Model(network, loss_fn, optimizer, metrics={'accuracy'})

    # 训练模型
    model.train(5, train_data, callbacks=[LossMonitor(1)], dataset_sink_mode=False)


    # 测试模型
    acc = model.eval(test_data, dataset_sink_mode=False)
    print(f"Test Accuracy: {acc['accuracy']:.4f}")

    return model, test_data

# 5. 可视化预测结果
def visualize_results(model, test_data):
    data = next(test_data.create_dict_iterator())
    images = data['image'].asnumpy()
    labels = data['label'].asnumpy()

    outputs = model.predict(Tensor(images))
    preds = outputs.argmax(axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("MNIST测试集预测结果", fontsize=16)

    for i in range(10):
        ax = axes[i // 5, i % 5]
        ax.imshow(images[i][0], cmap='gray')
        ax.set_title(f'Label: {labels[i]}\nPred: {preds[i]}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# 6. 主函数
model, test_data = train_model()
visualize_results(model, test_data)
