import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
lg=pickle.load(open('machine learning-deep learning/HTRU_2/deployment/webapp/model/lg.pkl','rb'))
knn=pickle.load(open('machine learning-deep learning/HTRU_2/deployment/webapp/model/knn.pkl','rb'))
svc=pickle.load(open('machine learning-deep learning/HTRU_2/deployment/webapp/model/svc.pkl','rb'))
dt=pickle.load(open('machine learning-deep learning/HTRU_2/deployment/webapp/model/dt.pkl','rb'))
rf=pickle.load(open('machine learning-deep learning/HTRU_2/deployment/webapp/model/rf.pkl','rb'))
bg=pickle.load(open('machine learning-deep learning/HTRU_2/deployment/webapp/model/bg.pkl','rb'))
ada=pickle.load(open('machine learning-deep learning/HTRU_2/deployment/webapp/model/ada.pkl','rb'))
gd=pickle.load(open('machine learning-deep learning/HTRU_2/deployment/webapp/model/gd.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    features=[float(x) for x in request.form.values()]
    final_features=[np.array(features)]
    i=1
    if request.method=="POST":
        i=request.form.get("choice")

        if i=="1":
            predict_lg(final_features[:8])
        elif i=="2":
            predict_knn(final_features[:8])
        elif i=="3":
            predict_svc(final_features[:8])
        elif i=="4":
            predict_dt(final_features[:8])
        elif i=="5":
            predict_rf(final_features[:8])
        elif i=="6":
            predict_bg(final_features[:8])
        elif i=="7":
            predict_ada(final_features[:8])
        elif i=="8":
            predict_gd(final_features[:8])
   

def predict_lg(final_features):
    #final_features=final_features[:8]
    prediction=lg.predict(final_features)
    y_probabilities_test=lg.predict_proba(final_features)
    y_prob_success=y_probabilities_test[:,1]
    print("final features",final_features)
    print("prediction:",prediction)
    output=round(prediction[0],2)
    y_prob=round(y_prob_success[0],3)
    y_prob*=100
    print(output)

    if output==0:
        return render_template('index.html',prediction_text='The star pulsar star with a probability of: {}%'.format(y_prob))
    else:
        return render_template('index.html',prediction_text='The star is not a pulsar star with a probability of: {}%'.format(y_prob))

def predict_knn(final_features):
    prediction=knn.predict(final_features)
    y_probabilities_test=knn.predict_proba(final_features)
    y_prob_success=y_probabilities_test[:,1]
    print("final features",final_features)
    print("prediction:",prediction)
    output=round(prediction[0],2)
    y_prob=round(y_prob_success[0],3)
    y_prob*=100
    print(output)

    if output==0:
        return render_template('index.html',prediction_text='The star pulsar star with a probability of: {}%'.format(y_prob))
    else:
        return render_template('index.html',prediction_text='The star is not a pulsar star with a probability of: {}%'.format(y_prob))

def predict_svc(final_features):
    prediction=svc.predict(final_features)
    y_probabilities_test=svc.predict_proba(final_features)
    y_prob_success=y_probabilities_test[:,1]
    print("final features",final_features)
    print("prediction:",prediction)
    output=round(prediction[0],2)
    y_prob=round(y_prob_success[0],3)
    y_prob*=100
    print(output)

    if output==0:
        return render_template('index.html',prediction_text='The star pulsar star with a probability of: {}%'.format(y_prob))
    else:
        return render_template('index.html',prediction_text='The star is not a pulsar star with a probability of: {}%'.format(y_prob))

def predict_dt(final_features):
    prediction=dt.predict(final_features)
    y_probabilities_test=dt.predict_proba(final_features)
    y_prob_success=y_probabilities_test[:,1]
    print("final features",final_features)
    print("prediction:",prediction)
    output=round(prediction[0],2)
    y_prob=round(y_prob_success[0],3)
    y_prob*=100
    print(output)

    if output==0:
        return render_template('index.html',prediction_text='The star pulsar star with a probability of: {}%'.format(y_prob))
    else:
        return render_template('index.html',prediction_text='The star is not a pulsar star with a probability of: {}%'.format(y_prob))

def predict_rf(final_features):
    prediction=rf.predict(final_features)
    y_probabilities_test=rf.predict_proba(final_features)
    y_prob_success=y_probabilities_test[:,1]
    print("final features",final_features)
    print("prediction:",prediction)
    output=round(prediction[0],2)
    y_prob=round(y_prob_success[0],3)
    y_prob*=100
    print(output)

    if output==0:
        return render_template('index.html',prediction_text='The star pulsar star with a probability of: {}%'.format(y_prob))
    else:
        return render_template('index.html',prediction_text='The star is not a pulsar star with a probability of: {}%'.format(y_prob))

def predict_bg(final_features):
    prediction=bg.predict(final_features)
    y_probabilities_test=bg.predict_proba(final_features)
    y_prob_success=y_probabilities_test[:,1]
    print("final features",final_features)
    print("prediction:",prediction)
    output=round(prediction[0],2)
    y_prob=round(y_prob_success[0],3)
    y_prob*=100
    print(output)

    if output==0:
        return render_template('index.html',prediction_text='The star pulsar star with a probability of: {}%'.format(y_prob))
    else:
        return render_template('index.html',prediction_text='The star is not a pulsar star with a probability of: {}%'.format(y_prob))

def predict_ada(final_features):
    prediction=ada.predict(final_features)
    y_probabilities_test=ada.predict_proba(final_features)
    y_prob_success=y_probabilities_test[:,1]
    print("final features",final_features)
    print("prediction:",prediction)
    output=round(prediction[0],2)
    y_prob=round(y_prob_success[0],3)
    y_prob*=100
    print(output)

    if output==0:
        return render_template('index.html',prediction_text='The star pulsar star with a probability of: {}%'.format(y_prob))
    else:
        return render_template('index.html',prediction_text='The star is not a pulsar star with a probability of: {}%'.format(y_prob))

def predict_gd(final_features):
    prediction=gd.predict(final_features)
    y_probabilities_test=gd.predict_proba(final_features)
    y_prob_success=y_probabilities_test[:,1]
    print("final features",final_features)
    print("prediction:",prediction)
    output=round(prediction[0],2)
    y_prob=round(y_prob_success[0],3)
    y_prob*=100
    print(output)

    if output==0:
        return render_template('index.html',prediction_text='The star pulsar star with a probability of: {}%'.format(y_prob))
    else:
        return render_template('index.html',prediction_text='The star is not a pulsar star with a probability of: {}%'.format(y_prob))


if __name__=="__main__":
    app.run(debug=True)