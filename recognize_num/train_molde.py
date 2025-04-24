import tensorflow as tf
import keras


# 定义模型
def create_molde():
    # 创建一个Sequential模型
    model = keras.Sequential()
    # 添加一个卷积层，32个3x3的卷积核，激活函数为relu
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # 添加一个最大池化层，2x2的池化窗口
    model.add(keras.layers.MaxPooling2D((2, 2)))
    # 添加一个卷积层，64个3x3的卷积核，激活函数为relu
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # 添加一个最大池化层，2x2的池化窗口
    model.add(keras.layers.MaxPooling2D((2, 2)))
    #添加全连接层
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    # 返回模型
    return model
#导入数据集
def input_data():
    # 加载MNIST数据集
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # 将训练集和测试集的形状调整为(-1, 28, 28, 1)
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    # 将训练集和测试集的数据类型转换为float32，并归一化到0-1之间
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # 将标签转换为one-hot编码
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    # 返回训练集和测试集的数据和标签
    return x_train, y_train, x_test, y_test

def model_train(model, x_train, y_train, x_test, y_test):
    # 编译模型，优化器为adam，损失函数为categorical_crossentropy，评估指标为accuracy
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # 训练模型，batch_size为128，训练10个epoch，验证集为测试集
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
    # 返回训练好的模型
    return model

# 导入数据集
train_data,train_target,test_data,test_target=input_data()
# 创建模型
model=create_molde()
# 训练模型
model=model_train(model,train_data,train_target,test_data,test_target)
# 保存模型
model.save('recognize_num.keras')