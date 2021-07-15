import flask
import pickle
import pandas as pd
import numpy as np


#load models at top of app to load into memory only one time
with open('models/default_lr.pkl', 'rb') as f:
    default_lr = pickle.load(f)

#feature space
df_train_jl_scale = pd.read_csv('data/df_train_jl_scale.csv')
df_features = pd.read_csv('data/features.csv')

sub_grade_dict={'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5,
          'B1': 6, 'B2': 7, 'B3': 8, 'B4': 9, 'B5': 10,
          'C1': 11, 'C2': 12, 'C3': 13, 'C4': 14, 'C5': 15,
          'D1': 16, 'D2': 17, 'D3': 18, 'D4': 19, 'D5': 20,
          'E1': 21, 'E2': 22, 'E3': 23, 'E4': 24, 'E5': 25,
          'F1': 26, 'F2': 27, 'F3': 28, 'F4': 29, 'F5': 30,
          'G1': 31, 'G2': 32, 'G3': 33, 'G4': 34, 'G5': 35,
         }
emp_length_dict={'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5,  
          '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10, 'unknown': 11}


home_to_int = {'MORTGAGE': 4,
               'RENT': 3,
               'OWN': 5,         
               'ANY': 2,            
               'OTHER': 1,          
               'NONE':0 }

purpose_dict = {"Credit Card": 'credit_card', "Debt Consolidation": "debt_consolidation",
                "Education":'education', "Home Improvement": "home_improvement", 
                'House':'house', "Major Purchase": "major_purchase", 'Medical':'medical',
                'Moving':'moving', 'Renewable Energy':'renewable_energy', 
                "Small Business": "small_business", 'Vacation': 'vacation',
                'Wedding':'wedding', 'Other':'other'}


application_type_dict = {'Joint App': 1, 'Individual': 0 }

app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return (flask.render_template('index.html'))

@app.route('/report')
def report():
    return (flask.render_template('report.html'))

@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    
    if flask.request.method == 'GET':
        return (flask.render_template('prediction.html'))
    
    if flask.request.method =='POST':
        
        #get input
        
        #sub-grade
        sub_grade = flask.request.form['sub_grade']
        #emp_length
        emp_length = flask.request.form['emp_length']
        #address state
        addr_state = flask.request.form['addr_state']
        #fico score as integer
        fico_avg = int(flask.request.form['fico_avg'])
        #loan amount as integer
        funded_amnt = float(flask.request.form['funded_amnt'])
        #debt to income as float
        dti = float(flask.request.form['dti'])
        #interests rate
        int_rate = float(flask.request.form['int_rate'])

        #bankrupcy records
        pub_rec_bankruptcies = int(flask.request.form['pub_rec_bankruptcies'])
        #home ownership as string
        home_ownership = flask.request.form['home_ownership']
        #purpose
        purpose = flask.request.form['purpose']
        #application type
        application_type = flask.request.form['application_type']
        #annual income as float
        annual_inc = float(flask.request.form['annual_inc'])
        #verification status as 0, 1, 2
        verification_status = flask.request.form['verification_status']
        #time since first credit line in months
        er_credit_open_date = pd.to_datetime(flask.request.form['er_credit_open_date'])
        cr_hist = 2020 - pd.to_datetime(er_credit_open_date).year

        
        
        
        #temp data frame
        data = np.array([np.zeros(78).astype(int)])
        temp = pd.DataFrame(data, columns=df_features.columns)
 
        temp['sub_grade']=sub_grade_dict[sub_grade]
        temp['emp_length']=emp_length_dict[emp_length]
        temp['addr_state_'+ addr_state]= 1
        temp['fico_avg'] = fico_avg
        temp['funded_amnt'] = np.log(funded_amnt)
        temp['dti']=dti
        temp['int_rate']=int_rate
        temp['pub_rec_bankruptcies']=pub_rec_bankruptcies
        temp['home_ownership_' + home_ownership] = 1
        temp['purpose_' + purpose_dict[purpose]] = 1
        temp['application_type_Joint App'] = application_type_dict[application_type]
        temp['annual_inc']=np.log(annual_inc)
        if verification_status in ['Source Verified', 'Verified']:
            temp['verification_status_' + verification_status]= 1
        temp['cr_hist']=cr_hist

        #create original output dict
        output_dict= dict()
        output_dict['Provided Annual Income'] = annual_inc
        output_dict['Provided FICO Score'] = fico_avg
        output_dict['Funded Amount']=funded_amnt
        
            
        #make prediction
        pred = default_lr.predict(temp)
        pred_proba = default_lr.predict_proba(temp)
        print('check:', pred_proba)
        
        res = ''
        if pred == 0:
            res = f'Everything looks good: {pred_proba}' 
        elif pred == 1:
            res = f'Default Warning: {pred_proba}'

 
        
        
        #render form again and add prediction
        return flask.render_template('prediction.html',
                                     original_input=output_dict,
                                     result=res,
                                     )
             
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)