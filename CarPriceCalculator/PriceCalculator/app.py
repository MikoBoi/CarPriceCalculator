from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

modelpkl = pickle.load(open('gboostmodel.pkl', 'rb'))
data = pd.read_csv("Cleaned Data.csv")

@app.route('/')
def index():
    marka = sorted(data['manufacturer'].unique())
    model = sorted(data['model'].unique())
    god = sorted(data['year'].unique(), reverse=True)
    toplivo = data['fuel'].unique()
    sostoianie = sorted(data['condition'].unique())

    return render_template('index.html', marka=marka, model=model, god=god, toplivo=toplivo, sostoianie=sostoianie)

@app.route('/predict', methods=['POST'])
def predict():
    manufact = request.form.get('marka')
    model = request.form.get('car_model')
    year = int(request.form.get('god'))
    fueltype = request.form.get('toplivo')
    status = request.form.get('sostoianie')
    kmstravelled = int(request.form.get('kms'))

    print(manufact, model, year, fueltype, kmstravelled)

    pred = modelpkl.predict(pd.DataFrame([[manufact, model, fueltype, status, year, kmstravelled]], columns=['manufacturer', 'model', 'fuel', 'condition', 'year', 'odometer']))

    print(pred)

    return str(pred[0])

if __name__ == '__main__':
    app.run(debug=True)
