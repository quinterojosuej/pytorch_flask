import io
import json

import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

import flask
from flask import Flask, jsonify, request, render_template, redirect

app = Flask(__name__)

# FLASK_ENV=development FLASK_APP=flaskSimple/app.py flask run
# to run the app you need to run from the environment
classify = ['BACTERIA', 'NORMAL', 'VIRUS']

def transform_image(image):
    data_transform = transforms.Compose([
    transforms.Resize((256, 192)), # images are different sizes, height x width
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    to_transform = Image.open(io.BytesIO(image))
    
    to_transform = to_transform.convert('RGB') # if image is grayscale
    return data_transform(to_transform).unsqueeze(0)

def get_prediction(image_bytes, classes=classify):
    tensor = transform_image(image=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted = int(y_hat.item())
#     print(classes[predicted])
    return classes[predicted] # returns the index!


device = torch.device("cpu")
model = torch.load('./model/resnet34_pneumonia_model.pt', map_location=device)
model.eval()


@app.route('/', methods=['GET', 'POST'])
def home():
    if flask.request.method == 'GET':
        return render_template('home.html')
    if flask.request.method == 'POST':
        file = request.files['file']
    
        if(file.filename.split('.')[-1].lower() != "jpeg" and file.filename.split('.')[-1].lower() != "jpg"): # checks if it is a jpeg
            print('not jpg or jpeg it was a %s' % (file.filename.split('.')[-1].lower()))
            return redirect('/')
        print(file.filename)

        image = file.read()
        class_name = get_prediction(image_bytes=image)
        print('got image!', class_name)
        return render_template('home.html', class_name=class_name)

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contatct():
    return render_template('contact.html')
    
# @app.route('/test', methods=['POST']) # tester!
# def test():
#     file = request.files['file']
    
#     if(file.filename.split('.')[-1].lower() != "jpeg" and file.filename.split('.')[-1].lower() != "jpg"): # checks if it a jpeg
#         print('not jpg or jpeg it was a %s' % (file.filename.split('.')[-1].lower()))
#         return redirect('/')
#     print(file.filename)
    
#     image = file.read()
#     class_name = get_prediction(image_bytes=image)
#     print('got image!')
#     return render_template('results.html', class_name=class_name)

# @app.route('/predict', methods=['POST']) # api in the futures are like this!!
# def predict():
#     if request.method == 'POST':
#         # get the file
#         file = request.files['file']
#         image = file.read()
#         class_name = get_prediction(image_bytes=image)
#         return jsonify({"class": class_name})


if __name__ == '__main__':
    app.run()