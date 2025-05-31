import os
from dash import Dash, dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Dash(__name__)
server = app.server

scaler = MinMaxScaler(feature_range=(0, 1))

df_nse = pd.read_csv("./NSE-Tata-Global-Beverages-Limited.csv")
df_nse["Date"] = pd.to_datetime(df_nse["Date"], format="%Y-%m-%d")
df_nse.index = df_nse['Date']

data = df_nse.sort_index(ascending=True, axis=0)

# Use vectorized assignment to avoid chained indexing issues
new_data = pd.DataFrame({
    "Date": data["Date"].values,
    "Close": data["Close"].values
})
new_data.index = new_data["Date"]
new_data.drop("Date", axis=1, inplace=True)

dataset = new_data.values

train = dataset[0:987, :]
valid = dataset[987:, :]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Check for the model file and load it
model_path = "saved_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the file exists or update the path.")

inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

# Create copies to avoid SettingWithCopy warnings
train_data = new_data[:987]
valid_data = new_data[987:].copy()
valid_data['Predictions'] = closing_price

df = pd.read_csv("./stock_data.csv")

app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            html.Div([
                html.H2("Actual closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=train_data.index,
                                y=valid_data["Close"],
                                mode='markers'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter Plot - Actual Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                ),
                html.H2("LSTM Predicted closing price", style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data": [
                            go.Scatter(
                                x=valid_data.index,
                                y=valid_data["Predictions"],
                                mode='markers'
                            )
                        ],
                        "layout": go.Layout(
                            title='Scatter Plot - Predicted Closing Price',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label='Facebook Stock Data', children=[
            html.Div([
                html.H1("Facebook Stocks High vs Lows", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='my-dropdown',
                    options=[
                        {'label': 'Tesla', 'value': 'TSLA'},
                        {'label': 'Apple', 'value': 'AAPL'},
                        {'label': 'Facebook', 'value': 'FB'},
                        {'label': 'Microsoft', 'value': 'MSFT'}
                    ],
                    multi=True, value=['FB'],
                    style={"display": "block", "margin-left": "auto",
                           "margin-right": "auto", "width": "60%"}
                ),
                dcc.Graph(id='highlow'),
                html.H1("Facebook Market Volume", style={'textAlign': 'center'}),
                dcc.Dropdown(
                    id='my-dropdown2',
                    options=[
                        {'label': 'Tesla', 'value': 'TSLA'},
                        {'label': 'Apple', 'value': 'AAPL'},
                        {'label': 'Facebook', 'value': 'FB'},
                        {'label': 'Microsoft', 'value': 'MSFT'}
                    ],
                    multi=True, value=['FB'],
                    style={"display": "block", "margin-left": "auto",
                           "margin-right": "auto", "width": "60%"}
                ),
                dcc.Graph(id='volume')
            ], className="container"),
        ])
    ])
])

@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft"}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["High"],
                mode='lines', opacity=0.7,
                name=f'High {dropdown[stock]}'
            )
        )
        trace2.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["Low"],
                mode='lines', opacity=0.6,
                name=f'Low {dropdown[stock]}'
            )
        )
    traces = trace1 + trace2
    figure = {
        'data': traces,
        'layout': go.Layout(
            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(dropdown[i] for i in selected_dropdown)} Over Time",
            xaxis={
                "title": "Date",
                'rangeselector': {'buttons': [
                    {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                    {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                    {'step': 'all'}
                ]},
                'rangeslider': {'visible': True},
                'type': 'date'
            },
            yaxis={"title": "Price (USD)"}
        )
    }
    return figure

@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph_volume(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft"}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["Volume"],
                mode='lines', opacity=0.7,
                name=f'Volume {dropdown[stock]}'
            )
        )
    figure = {
        'data': trace1,
        'layout': go.Layout(
            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(dropdown[i] for i in selected_dropdown_value)} Over Time",
            xaxis={
                "title": "Date",
                'rangeselector': {'buttons': [
                    {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                    {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                    {'step': 'all'}
                ]},
                'rangeslider': {'visible': True},
                'type': 'date'
            },
            yaxis={"title": "Transactions Volume"}
        )
    }
    return figure

if __name__ == '__main__':
    app.run(debug=True)
