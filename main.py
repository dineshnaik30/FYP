from flask import Flask, redirect, url_for, request,render_template
import joblib
import json

heartModel = joblib.load('svm_heart_model.pkl')
diabetesModel = joblib.load('svm_diabetes_model.pkl')
healthModel = joblib.load('svm_health_model.pkl')

app = Flask(__name__)


@app.route("/")
def home():
   url_for('static',filename='templates/css/app.9807e8f4.css')
   return render_template("index.html")


@app.route('/sendData',methods = ['POST'])
def login():
   if request.method == 'POST':
      data = request.json
      heartResult = ["0"]
      diabetsResult = ["0"]
      healthResult = ["0"]

      heartResult = heartModel.predict([[int(data['Age']),int(data['Gender']),int(data['cp']),int(data['SBP']),int(data['fbs']),int(data['restecg']),int(data['HeartRate'])]])      
      diabetsResult = diabetesModel.predict([[int(data['Pregnancies']),int(data['Glucose']),int(data['DBP']),float(data['Bodymass']),int(data['Age'])]])
      healthResult = healthModel.predict([[float(data['Bodymass']),float(data['SBP']),float(data['BodyTemp']),int(data['Glucose']),int(data['HeartRate'])]])

      result = {
         'heartResult' : str(heartResult[0]),
         'diabetesResult' : str(diabetsResult[0]),
         'healthResult' : str(healthResult[0])
      }
      print(result)
      return json.dumps(result)
   

if __name__ == '__main__':
   app.run(debug = True)