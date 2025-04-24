import tensorflow as tf
import keras
from PIL import Image
import numpy as np

# 加载训练好的模型
num_model = keras.models.load_model('recognize_num.keras')

# 打开图片
img = Image.open('2.jpg')
# 将图片大小调整为28*28
img = img.resize((28, 28))
# 将图片转换为灰度图
img = img.convert('L')
# 将图片转换为numpy数组
img = np.array(img)
# 将图片像素值归一化到0-1之间
img = (255-img)/255
# 将图片形状调整为(1, 28, 28, 1)，其中1表示图片数量，28*28表示图片大小，1表示图片通道数
img = img.reshape((1, 28, 28, 1))

# 使用模型进行预测
prediction = num_model.predict(img)
print(prediction)
# 获取预测结果中概率最大的类别
prediction = np.argmax(prediction)
print(prediction)