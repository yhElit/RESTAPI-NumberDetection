from flask import Flask, request, jsonify
from sklearn import datasets, svm
from joblib import dump, load
import numpy as np
import cv2
import os

app = Flask(__name__)


def create_opencv_image_from_stringio(img_stream, cv2_img_flag=0):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)


@app.route('/')
def index():
    return 'Server Works!'


@app.route('/swe/numberprediction/rest/data/v1.0/json/en/<model>', methods=['DELETE'])
def delete_model(model):
    try:
        os.remove(model)
        return "model: " + model + "successfully deleted."
    except:
        return "No such model exists"


@app.route('/swe/numberprediction/rest/data/v1.0/json/en/<model>', methods=['POST'])
def create_model(model):
    digits = datasets.load_digits()

    # Create features and targets
    x = digits.data
    y = digits.target

    # Classifier implementing the k-nearest neighbors vote and support vector machine
    clf_svm = svm.SVC(gamma=0.001, probability=True)

    # Fit the knn and svc classifier from the training dataset
    clf_svm.fit(x, y)
    dump(clf_svm, model)

    return "model: " + model + " saved."


@app.route('/swe/numberprediction/rest/data/v1.0/json/en/<model>', methods=['GET'])
def get_prediction(model):
    filter = request.args.get("filter")
    if filter != "what_is_this_number" and filter != "what_is_this_numbers_probabilities" and filter != "what_is_this_number_probability":
        return "Wrong filter"

    try:
        image = request.files['file']
    except:
        return "There was a problem with the image"

    try:
        m_model = load(model)
    except:
        return "No such model exists"

    im = create_opencv_image_from_stringio(image, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(im, (8, 8))

    # Invert Image to white number with black background
    image_resized = cv2.bitwise_not(image_resized)

    # Normalize data to fit model from 0-255 to 0-16
    oldRange = 255
    newRange = 16
    a = [((((jj - 0) * newRange) / oldRange) + 0).astype(int) for j in image_resized for jj in j]
    a = np.array(a).reshape(1, -1)

    # Predict Number
    # prediction = model.predict(a)
    prediction_prob = m_model.predict_proba(a)

    result = {}
    print_result = ""

    for x in m_model.classes_:
        tmp = prediction_prob[0][x] * 100
        print_result = print_result + "Number " + str(x) + ": " + str(tmp) + " % "
        result[x] = tmp

    prediction = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    predicted_number = str(list(prediction.items())[0][0])
    predicted_probability = str(list(prediction.items())[0][1])

    if filter == "what_is_this_number":
        return jsonify("predicted number is " + predicted_number)
    if filter == "what_is_this_number_probability":
        return jsonify("The image is with " + predicted_probability + " probability Number " + predicted_number)
    if filter == "what_is_this_numbers_probabilities":
        return jsonify(str(print_result))
