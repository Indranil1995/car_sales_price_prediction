from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        km_driven=int(request.form['km_driven'])
        km_driven2=np.log(km_driven)
        no_year=int(request.form['no_year'])
        fuel_type=request.form['fuel_type']
        if(fuel_type=='Petrol'):
                fuel_type=1
                fuel_type=0
        else:
            fuel_type=0
            fuel_type=1
        seller_type=request.form['seller_type']
        if(seller_type=='Individual'):
            seller_type=1
            seller_type=0
        else:
            seller_type=0
            seller_type=1	
        Transmission_Mannual=request.form['Transmission_Mannual']
        if(Transmission_Mannual=='Mannual'):
            Transmission_Mannual=1
        else:
            Transmission_Mannual=0
        Owner=int(request.form['Owner'])    
        prediction=model.predict([[km_driven2,no_year,fuel_type,seller_type,Transmission_Mannual,Owner]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)


