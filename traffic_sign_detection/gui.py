from keras.models import load_model
from PIL import ImageTk
import PIL.Image
from tkinter import filedialog
import customtkinter
from tkinter import *
import tkinter as tk
import numpy as np
import os.path
import cv2


path = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\traffic_sign_detection\archive\Output'
model_path = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('hdf5')][0]

model = load_model(model_path)

classes = {1: 'Speed limit (20km/h)',
           2: 'Speed limit (30km/h)',
           3: 'Speed limit (50km/h)',
           4: 'Speed limit (60km/h)',
           5: 'Speed limit (70km/h)',
           6: 'Speed limit (80km/h)',
           7: 'End of speed limit (80km/h)',
           8: 'Speed limit (100km/h)',
           9: 'Speed limit (120km/h)',
           10: 'No passing',
           11: 'No passing veh over 3.5 tons',
           12: 'Right-of-way at intersection',
           13: 'Priority road',
           14: 'Yield',
           15: 'Stop',
           16: 'No vehicles',
           17: 'Veh > 3.5 tons prohibited',
           18: 'No entry',
           19: 'General caution',
           20: 'Dangerous curve left',
           21: 'Dangerous curve right',
           22: 'Double curve',
           23: 'Bumpy road',
           24: 'Slippery road',
           25: 'Road narrows on the right',
           26: 'Road work',
           27: 'Traffic signals',
           28: 'Pedestrians',
           29: 'Children crossing',
           30: 'Bicycles crossing',
           31: 'Beware of ice/snow',
           32: 'Wild animals crossing',
           33: 'End speed + passing limits',
           34: 'Turn right ahead',
           35: 'Turn left ahead',
           36: 'Ahead only',
           37: 'Go straight or right',
           38: 'Go straight or left',
           39: 'Keep right',
           40: 'Keep left',
           41: 'Roundabout mandatory',
           42: 'End of no passing',
           43: 'End no passing veh > 3.5 tons'}

customtkinter.set_appearance_mode('system')
customtkinter.set_default_color_theme('blue')

root = customtkinter.CTk()
root.geometry('800x600')
root.title('Traffic Sign Classification')

sign_image = customtkinter.CTkLabel(root)
label = customtkinter.CTkLabel(root, text_font=('arial', 15))


def classify(file_path):
    global label_packed
    image = cv2.imread(file_path)
    image = cv2.resize(image, (30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    predictions = model.predict(image)[0]
    idx = np.argmax(predictions, axis=-1)
    predictions = round(predictions[idx] * 100, 2)
    sign = classes[idx+1]
    label.configure(text=f'Label: {sign}\nProbability: {predictions}%')


def show_classify_button(file_path):
    classify_b = customtkinter.CTkButton(root, text='Classify Image', command=lambda: classify(file_path))
    classify_b.configure(fg_color='white')
    classify_b.place(relx=0.5, rely=0.5)


def upload_images():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = PIL.Image.open(file_path)
        uploaded = uploaded.resize((250, 250))
        img = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=img)
        sign_image.image = img
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = customtkinter.CTkButton(root, text='Upload an image', command=upload_images)
upload.configure(fg_color='black')
upload.pack(side=BOTTOM, pady=50)

sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = customtkinter.CTkLabel(root, text='Traffic Sign Image', text_font=('arial', 20))
heading.pack()

root.mainloop()
