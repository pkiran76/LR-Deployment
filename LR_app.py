from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler

model1=pd.read_pickle("admission.pkl")
scale=pd.read_pickle("scaling.pkl")
#Import the pre-processor separately to ensure that the same object of the class is used to transform
# the new data

app=Flask(__name__)


@app.route("/",methods=["GET","POST"])
def home_page():
    return render_template("index.html")

@app.route("/math",methods=["POST"])
def admission():
    if (request.method=='POST'):
        GRE_Score=int(request.form['GRE Score'])
        TOEFL_Score=int(request.form['TOEFL Score'])
        University_Rating_Score=int(request.form['University Rating Score'])
        SOP=float(request.form['SOP'])
        LOR=float(request.form['LOR'])
        CGPA=float(request.form['CGPA'])
        Research=int(request.form['Research'])
        test1=scale.transform([[GRE_Score,TOEFL_Score,University_Rating_Score,SOP,LOR,CGPA,Research]])
        result=model1.predict(test1)*100
        return render_template('results.html', result=result)

if __name__ == '__main__':   #revoking the main object/constructor defined in the app object above
    app.run()