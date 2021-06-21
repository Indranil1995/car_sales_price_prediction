from re import X
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
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0],2)
    if output<0:
         return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
    else:
        return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))


if __name__=="__main__":
    app.run(debug=True)


