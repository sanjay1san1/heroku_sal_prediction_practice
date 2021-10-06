from flask import Flask,render_template,request
import joblib
import numpy as np
app = Flask(__name__)

#Load the model 
model = joblib.load('hiring_model.pkl') 

@app.route('/')
def hello():
    return render_template('base.html')

@app.route('/predict',methods = ['POST'])
def predict():
    exp=request.form.get('Experience')
    Test_Score=exp=request.form.get('Test_Score')
    Interview_Score=exp=request.form.get('Interview_Score')
    print('Your Experience' , exp)
    print('Your Test Score is ' , Test_Score)
    print('Your Interview Score is' , Interview_Score)

    prediction = model.predict([[int(exp),int(Test_Score),int(Interview_Score)]])  # Model prediction based on user input data in browser 
    print('Model prediction is : ' , prediction)
    output = round(prediction[0],2)

    #return 'Your form submitted successfull'
    return render_template('base.html',prediction_text = f'Employee salary will be $ {output}')
print( __name__)
if __name__ == '__main__':   
    app.run(debug=True)