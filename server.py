import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import cv2

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from testing import Testing

test=Testing()
app = Flask(__name__)
filename_global=""

@app.route('/', methods=['GET', 'POST'])
def main_page():
    print(request)
    try:
        if request.method == 'POST':
            file = request.files['file']
            filename = secure_filename(file.filename)
            file.save(os.path.join('uploads', filename))
            return redirect(url_for('prediction', filename=filename))
    except:
        pass
    return render_template('index.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    global filename_global
    filename_global=filename
    # Step 1
    my_image =cv2.imread(os.path.join("uploads",filename))
    # Step 2
    my_image_re = cv2.resize(my_image, (299,299))
    my_image_re=(my_image_re.astype('float16'))/255.0
    # Prediction
    prediction=test.predict(my_image_re)
    print(prediction)
    # Order the lists by decreasing probability
    zipped_lists = list(zip(prediction[0], prediction[1][0]))
    sorted_pairs = sorted(zipped_lists, key = lambda x: x[1])
    tuples = zip(*sorted_pairs)
    list1, list2 = [ list(tuple) for tuple in tuples]
    print(list1)
    print(list2)
    predictions = {
        "class1":list1[2],
        "class2":list1[1],
        "class3":list1[0],
        "prob1":list2[2],
        "prob2":list2[1],
        "prob3":list2[0],
      }
    return render_template('prediction.html', predictions=predictions)

@app.route("/partialFit/", methods=['GET','POST'])
def move_forward():
    print(filename_global)
    my_image =cv2.imread(os.path.join("uploads",filename_global))
    #Step 2
    my_image_re = cv2.resize(my_image, (299,299))
    my_image_re=(my_image_re.astype('float16'))/255.0
    y=request.args.get("param")
    test.partial_fit(my_image_re,y)
    return render_template('ty.html')


app.run(debug=False)