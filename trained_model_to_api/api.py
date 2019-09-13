from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["DEBUG"] = True

filename = 'trained_model.sav'
@app.route('/api/v0/validate', methods=['GET', 'POST'])
def validate():
    vector = pd.DataFrame([request.json])
    # use joblib to avoid stocking
    loaded_model = joblib.load(filename)
    res = str(loaded_model.predict(vector)[0])
    result=[
        {
            'id':0,
            'prediction':res
        }
    ]
    return jsonify(result)


