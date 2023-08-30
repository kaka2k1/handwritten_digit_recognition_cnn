import tensorflow as tf
import numpy as np
from PIL import Image

# Đường dẫn đến tập tin mô hình
model_path = 'best_model.h5'

# Tải mô hình đã lưu
model = tf.keras.models.load_model(model_path)

# Đường dẫn đến hình ảnh của bạn
image_path = 'image_test.png'

# Xử lý hình ảnh
image = Image.open(image_path).convert('L')
image = image.resize((28, 28))
image_array = np.array(image)
image_array = image_array.reshape((1, 28, 28, 1))
image_array = image_array.astype('float32') / 255

# Dự đoán nhãn của hình ảnh
predictions = model.predict(image_array)
predicted_label = np.argmax(predictions)

print("Predicted label:", predicted_label)
