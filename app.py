import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st
from mpl_finance import candlestick_ohlc
import datetime
import matplotlib.dates as mpl_dates
import plotly.graph_objects as pl

start = "2010-01-01"
end = "2022-12-31"

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'TSLA')
# df = yf.download(user_input, start=start, end=end)
df = yf.download(user_input, start=start)

#Describing Data
st.subheader('Stock Price Data')
st.write(df.tail())
st.write(df.describe())


df['Date'] = df.index
#Plotting Candlestick Chart
st.subheader('CandleStick Chart')
candlestick = pl.Candlestick(x=df['Date'],low=df['Low'],high=df['High'],close=df['Close'],open=df['Open'])
fig = pl.Figure(data=[candlestick])
fig.update_layout(
    width=800, height=700,
    # title="Apple, March - 2020",
    yaxis_title='Stock Price'
)

# plt.show() 
st.plotly_chart(fig)

#Visualization
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart With Moving Averages')
ma1 = st.number_input('Moving Average 1 Length',min_value=1, max_value=200, step=1, value=50)
ma2 = st.number_input('Moving Average 2 Length',min_value=1, max_value=200, step=1, value=50)
ma100 = df.Close.rolling(ma1).mean()
ma200 = df.Close.rolling(ma2).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

#Splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# print(data_training.shape)
# print(data_testing.shape)

# We will scale the data.(b/w 0 and 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Load my model
model = load_model('keras_model.h5')

#Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler1 = scaler.scale_

scale_factor = 1/scaler1[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Predictions Vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b',label = 'Original Price')
plt.plot(y_predicted, 'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#Next Day Predicted Price
realdata = pd.DataFrame(df['Close'].tail(100))
realdata_arr = np.array(realdata)
realdata_arr = scaler.transform(realdata_arr)
realdata_arr = np.reshape(realdata_arr, (1, realdata_arr.shape[0], 1))
pred=model.predict(realdata_arr)
pred = scaler.inverse_transform(pred)
pred=np.array(pred)
predi=pred[-1][0]
st.subheader(f"Next Day Predicted Price: {predi}")

