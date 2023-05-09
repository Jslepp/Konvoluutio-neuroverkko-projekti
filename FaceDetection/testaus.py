import os
import tkinter as tk
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
from keras.models import load_model

model = load_model('face_detection_model.h5')
test_dir = 'pictures/testing'
class_names = ['Face', 'Non-face']


correct_predictions = 0
total_images = 0
current_image_index = 0
images = []


window = tk.Tk()


canvas = tk.Canvas(window, width=180, height=180)
canvas.pack()


prediction_label = tk.Label(window, text="")
prediction_label.pack()


next_image_button = tk.Button(window, text="Next Image")


# Vastaa siirtymisestä kuvien välillä
def load_next_image():
    global images
    global correct_predictions
    global total_images
    global current_image_index
    
    
    if current_image_index < len(images):
        img = images[current_image_index]
        img = img.resize((180, 180))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) 
        predict(img_array,img)
        current_image_index += 1
        
# Käyttää luotua mallia arvioimaan onko kuvassa kasvoja
def predict(image_array,img):
    predictions = model.predict(image_array)
    score = tf.nn.softmax(predictions[0])
    prediction = class_names[np.argmax(score)]
    prediction_label.config(text="This image most likely contains a {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
        
    canvas.img = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor='nw', image=canvas.img)
    
# Luo listan johon tulee kuvat testing kansiosta
def load_test_images():
    images = []
    root_dir = os.getcwd()
    test_dir = os.path.join(root_dir, 'pictures', 'testing')
    for file in os.listdir(test_dir):
        if file.endswith('.jpg'):
            img = Image.open(os.path.join(test_dir, file))
            images.append(img)
    return images


next_image_button.config(command=load_next_image)

images = load_test_images()
load_next_image()


next_image_button.pack()

window.mainloop()