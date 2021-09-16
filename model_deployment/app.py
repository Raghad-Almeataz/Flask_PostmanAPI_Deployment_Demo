import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def home():
    if (request.method == 'GET'):
        data = 'hello world!'
        return jsonify({'data':data})
    
@app.route('/predict/')
def predict():
    model = pickle.load(open('pipemodel.pkl', 'rb'))
    age=request.args.get('age')
    sex=request.args.get('sex')
    cp=request.args.get('cp')
    trestbps=request.args.get('trestbps')
    chol=request.args.get('chol')
    tempdf={'age':[age], 'sex':[sex], 'cp':[cp], 
            'trestbps':[trestbps], 'chol':[chol]}
    test_df = pd.DataFrame(tempdf)
    status = model.predict(test_df)
    return jsonify({'Status':str(status)})
    
if __name__ == "__main__":
    app.run(debug=True)