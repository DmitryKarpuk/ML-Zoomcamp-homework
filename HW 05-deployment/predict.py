import pickle

def predict():
    test_in = {"reports": 0, "share": 0.001694,
               "expenditure": 0.12, "owner": "yes"}
    files_ls = ['dv.bin', 'model1.bin']
    loads = []
    for file in files_ls:
        with open(file, 'rb') as f_in:
            loads.append(pickle.load(f_in))
    dv = loads[0]
    model = loads[1]
    print('Probability: ',model.predict_proba(
                    dv.transform(test_in))[0][1])

if __name__ == "__main__":
    predict()