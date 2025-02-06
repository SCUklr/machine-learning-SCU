from libsvm.svmutil import *
import numpy as np

# 数据集
X = [
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318],
    [0.556, 0.215], [0.403, 0.237], [0.481, 0.149], [0.437, 0.211],
    [0.666, 0.091], [0.243, 0.267], [0.245, 0.057], [0.343, 0.099],
    [0.639, 0.161], [0.657, 0.198], [0.360, 0.370], [0.593, 0.042],
    [0.719, 0.103]
]
y = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]

# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 转换为 LIBSVM 格式
problem = svm_problem(y, X_scaled.tolist())

# 线性核 SVM 模型
linear_param = svm_parameter('-t 0 -c 1')  # -t 0 表示线性核
linear_model = svm_train(problem, linear_param)

# 高斯核 SVM 模型
rbf_param = svm_parameter('-t 2 -c 1 -g 0.5')  # -t 2 表示RBF核，-g 指定 gamma
rbf_model = svm_train(problem, rbf_param)

# 支持向量分析
print("Linear Kernel SVM Support Vectors:")
print(np.array(linear_model.get_SV()))  # 打印线性核支持向量

print("RBF Kernel SVM Support Vectors:")
print(np.array(rbf_model.get_SV()))  # 打印RBF核支持向量

