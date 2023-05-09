import cv2
import glob
import os
import time
from tkinter import *
from PIL import Image, ImageTk

image_files = glob.glob("pictures/testing/*.jpg")
root = Tk()
root.geometry("500x500")
image_label = Label(root)
image_label.pack()

next_button = Button(root, text="Next")

image_index = 0


def show_next_image():
    global image_index
    if image_index < len(image_files):
        
        image_path = image_files[image_index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            

        height, width, channels = image.shape
        max_height = root.winfo_screenheight() - 100
        max_width = root.winfo_screenwidth() - 100
        if height > max_height:
            scale = max_height / height
            height = int(max_height)
            width = int(width * scale)
        if width > max_width:
            scale = max_width / width
            width = int(max_width)
            height = int(height * scale)
        image = cv2.resize(image, (width, height))
        
        
        photo = ImageTk.PhotoImage(Image.fromarray(image))
        image_label.config(image=photo)
        image_label.image = photo

        image_index += 1
        



next_button.config(command=show_next_image)
next_button.pack()


show_next_image()
root.mainloop()