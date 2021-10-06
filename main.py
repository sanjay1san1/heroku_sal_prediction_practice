from flask import Flask,render_template,request
import joblib
app = Flask(__name__)
model = joblib.load(filename='hiring_model.pkl')
@app.route('/')
def hello():
    return render_template('base2.html')
@app.route('/predict',methods = ['POST'])
def predict():
    exp=request.form.get('Experience')
    t_score=request.form.get('Test_Score')
    I_score=request.form.get('Interview_Score')
    print('Your Experience : ',exp)
    print('Your Test_Score : ',t_score)
    print('Your Interview_Score : ',I_score)
    prediction=model.predict([[int(exp),int(t_score),int(I_score)]])
    result = round(prediction[0],2)
    print('model prediction for User Data' , result)
    return render_template('base2.html',prediction_text = f'Employee salary will be ${result}')
app.run(debug=True)