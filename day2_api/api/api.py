import joblib
import pandas as pd
from flask import Flask, request, jsonify
import json

# To run the API, run `flask --app api --debug run`
# Go to http://127.0.0.1:5000/api/gender to see the result

app = Flask(__name__)

GENDERS = {1: "M", 2: "F"}

def vectorize(name):
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    return pd.Series([int(letter in name.lower()) for letter in alphabet])


@app.route("/api/gender")
def gender():
    name = request.args.get('name', None)

    if name is None:
        return "Please provide a name", 400

    regr = joblib.load("api/model.prenoms.bin")
    vector = vectorize(name)
    prediction = regr.predict([vector])[0]
    gender = GENDERS[prediction]

    return jsonify({
        "name": name,
        "gender": gender,
    })