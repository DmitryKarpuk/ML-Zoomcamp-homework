import pickle
from flask import Flask, request, jsonify


#load data
files_ls = ['dv.bin', 'model1.bin']
loads = []
for file in files_ls:
    with open(file, 'rb') as f_in:
        loads.append(pickle.load(f_in))
dv = loads[0]
model = loads[1]

app = Flask('credict_card')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    prob = model.predict_proba(X)[0][1]
    result = {'card probabilities' : float(prob)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)