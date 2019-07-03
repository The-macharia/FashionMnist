import numpy as np
from PIL import Image
from keras.models import load_model

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
model = load_model('Trained_model2.h5')

input_path = input("Enter an image file path: ")
input_image = Image.open(input_path)
input_image = input_image.resize((28, 28), resample=Image.LANCZOS)

image_array = np.array(input_image)
image_array = image_array.astype('float32')
image_array /= 255
image_array = image_array.reshape(1, (28, 28))

answer = model.predict(image_array)
input_image.show()
print(class_names[np.argmax(answer)])
