import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)
model=pickle.load(open('machine learning-deep learning/breast cancer prediction/deployment/webapp/model/model.pkl','rb'))
scaler=pickle.load(open('machine learning-deep learning/breast cancer prediction/deployment/webapp/model/scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    features=[float(x) for x in request.form.values()]
    final_features=[np.array(features)]
    final_features=scaler.transform(final_features)
    prediction=model.predict(final_features)
    y_probabilities_test=model.predict_proba(final_features)
    y_prob_success=y_probabilities_test[:,1]
    print("final features",final_features)
    print("prediction:",prediction)
    output=round(prediction[0],2)
    y_prob=round(y_prob_success[0],3)
    y_prob*=100
    print(output)

    if output==0:
        return render_template('index.html',prediction_text='The patient is likely to have benign tumour with probability of: {}%'.format(y_prob))
    else:
        return render_template('index.html',prediction_text='The patient is likely to have malignant tumour with probability of: {}%'.format(y_prob))
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    predict=model.predict([np.array(list(data.values()))])

    output=prediction[0]
    return jsonify(output)

if __name__=="__main__":
    app.run(debug=True)

    
