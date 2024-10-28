from flask import Flask, render_template, request
import yfinance as yf
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/templates/templates/index.html')
def home1():
    return render_template('index.html')

@app.route('/templates/index.html')
def home3():
    return render_template('index.html')

@app.route('/finallogin.html')
def login():
    return render_template('finallogin.html')

@app.route('/finalsignup.html')
def login1():
    return render_template('finalsignup.html')


@app.route('/topgain.html')
def topg():
    return render_template('topgain.html')

@app.route('/info.html')
def topg1():
    return render_template('info.html')

@app.route('/contect.html')
def topg2():
    return render_template('contect.html')

@app.route('/aboutus.html')
def topg3():
    return render_template('aboutus.html')

@app.route('/stock_data.html', methods=['POST'])
def stock_data():
    stock_name = request.form['stock_name']+'.NS'
    start_date = '2024-07-28'
    end_date = datetime.now().strftime('%Y-%m-%d')
    stock_data = yf.download(stock_name, start=start_date, end=end_date)
    stock_info = yf.Ticker(stock_name).info
    #stock_history = yf.Ticker(stock_symbol).history(start=start_date, end=end_date)
    #stock_prices = stock_data[['Open', 'High', 'Low', 'Close']]
    stock_prices = stock_data.reset_index().to_dict(orient='records')
    return render_template('stock_data.html', stock_info=stock_info, stock_prices=stock_prices)

@app.route('/templates/index1234.html', methods=['POST'])
def tool():
    return render_template('index1234.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.form['stock_name'] + '.NS'
    data = yf.download(tickers=stock_name,period='5y',interval='1d')
    opn = data[['Open']]
    ds = opn.values
    normalizer = MinMaxScaler(feature_range=(0,1))
    ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))
    len(ds_scaled), len(ds) 
    #Defining test and train data sizes
    train_size = int(len(ds_scaled)*0.70)
    test_size = len(ds_scaled) - train_size
    train_size,test_size
    #Splitting data between train and test
    ds_train, ds_test = ds_scaled[0:train_size,:], ds_scaled[train_size:len(ds_scaled),:1]
    len(ds_train),len(ds_test)
    #creating dataset in time series for LSTM model 
    #X[100,120,140,160,180] : Y[200]
    def create_ds(dataset,step):
        Xtrain, Ytrain = [], []
        for i in range(len(dataset)-step-1):
           a = dataset[i:(i+step), 0]
           Xtrain.append(a)
           Ytrain.append(dataset[i + step, 0])
        return np.array(Xtrain), np.array(Ytrain)
    #Taking 100 days price as one record for training
    time_stamp = 100
    X_train, y_train = create_ds(ds_train,time_stamp)
    X_test, y_test = create_ds(ds_test,time_stamp)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    model = Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1,activation='linear'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=60,batch_size=12)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = normalizer.inverse_transform(train_predict)
    test_predict = normalizer.inverse_transform(test_predict)

    if len(ds_test) >= 371:  # Ensure there's enough data for 270 + 101
        fut_inp = ds_test[270:]
    else:
        raise ValueError("Insufficient data in ds_test for starting point at index 270.")

    fut_inp = fut_inp.reshape(1, -1)
    tmp_inp = fut_inp.flatten().tolist()  # Ensures tmp_inp is a 1D list
    lst_output = []
    n_steps = 101
    i = 0

    while(i<2):
        if(len(tmp_inp)>101):
            fut_inp = np.array(tmp_inp[1:])
            fut_inp=fut_inp.reshape(1,-1)
            fut_inp = fut_inp.reshape((1, n_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            tmp_inp = tmp_inp[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            fut_inp = fut_inp.reshape((1, n_steps,1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1

    ds_new = ds_scaled.tolist()
    len(ds_new)
    ds_new.extend(lst_output)
    final_graph = normalizer.inverse_transform(ds_new).tolist()
    next_day = round(float(*final_graph[len(final_graph)-1]), 2)
    return render_template('index1234.html', next_day=next_day,stock_name=stock_name)

def create_ds(dataset,step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)

if __name__ == '__main__':
    app.run()                   #app.run(prot=8000);



