# Import Dependencies
import os
import pymongo
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bson import json_util
import Forecast
from pymongo import MongoClient
from tensorflow.keras.losses import MeanSquaredError
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from pickle import load
import tensorflow as tf

app = FastAPI()

# Set up CORS (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")

# Create connection to the database
client = MongoClient(MONGODB_URI)

# Select database
db = client.get_database(DB_NAME)

# SOLAR collection
solar_collection = db.solar_data
solar_df = pd.DataFrame(list(solar_collection.find())).drop('_id', axis=1)

# WIND collection
wind_collection = db.wind_data
wind_df = pd.DataFrame(list(wind_collection.find())).drop('_id', axis=1)

@app.get("/getSolar")
def get_solar_data():
    lat = "30.238333"
    lon = "-97.508611"

    # Make API call to get weather data and turn response to JSON object
    response_json = Forecast.makeAPIRequest(lat, lon, os.getenv('weather_api_key'))
    print(response_json)

    # Get the daily weather variables from json response and store as pandas dataframe
    daily_solar_df = Forecast.forecasted_daily_solar(response_json)
    
    num_days = len(daily_solar_df) 

    daily_index = np.arange(0, num_days, 1) 
    daily_utc_date_time = daily_solar_df["UTC_Time"]
    date_time_df = Forecast.convert_DateTime_UTC_to_CST(daily_utc_date_time, daily_index)

    # Calculate sunhours
    UTC_sunrise = daily_solar_df["Sunrise"]
    UTC_sunset = daily_solar_df["Sunset"]
    sunhour_df = Forecast.calculate_sunhour(UTC_sunrise, UTC_sunset, daily_index)

    daily_weather_df = pd.merge(daily_solar_df, date_time_df, on='UTC_Time', how='outer')
    daily_weather_df.drop(columns=["UTC_Time"], axis=1, inplace=True)

    daily_weather_df = pd.merge(daily_weather_df, sunhour_df, on='Sunrise', how='outer')
    daily_weather_df.drop(columns=["Sunrise", "Sunset", "Date_Time", "Year", "Month", "Hour"], axis=1, inplace=True)

    hourly_solar_df = Forecast.forecasted_hourly_solar(response_json)

    n_days = len(response_json["days"])

    total_hours = n_days * 24

    hourly_index = np.arange(0, total_hours, 1)
    hourly_utc_date_time = hourly_solar_df["UTC_Time"]
    hourly_date_time_df = Forecast.convert_DateTime_UTC_to_CST(hourly_utc_date_time, hourly_index)

    hourly_weather_df = pd.merge(hourly_solar_df, hourly_date_time_df, on='UTC_Time', how='outer')
    hourly_weather_df.drop(columns=["UTC_Time"], axis=1, inplace=True)
    hourly_weather_df["Date_Time"] = pd.to_datetime(hourly_weather_df["Date_Time"])

    forecasted_solar_df = pd.merge(hourly_weather_df, daily_weather_df, on='Day', how='inner')
    forecasted_solar_df = forecasted_solar_df[["Date_Time", "Year", "Month", "Day", "Hour", "Temperature_F", "Humidity_percent", "Sunhour", "CloudCover_percent", "uvIndex", "Weather_Description"]]

    # Load the model
    scaler = load(open(r'C:\Users\admin\Desktop\austin-green-energy-predictor-master\Solar\solar_ml_model\scaler.pkl', 'rb'))
    load_nn = tf.keras.models.load_model(r'C:\Users\admin\Desktop\austin-green-energy-predictor-master\Solar\solar_ml_model\solar_model.h5', custom_objects={'mse': MeanSquaredError()})

    # Define the features (X) and transform the data
    X = forecasted_solar_df.drop(['Date_Time', 'Weather_Description', 'Day', 'Year'], axis=1)
    X_scaled = scaler.transform(X)

    y_pred = load_nn.predict(X_scaled)
    y_pred = y_pred.ravel()
    forecasted_solar_df['pred'] = y_pred

    return forecasted_solar_df.to_json(orient='records', index=False)

@app.get("/getwind")
def get_wind_data():
    lat = "32.776111"
    lon = "-99.476444"

    response_json = Forecast.makeAPIRequest(lat, lon, os.environ.get('weather_api_key'))
    print(response_json)
    hourly_wind_df = Forecast.forecasted_hourly_wind(response_json)
    print(hourly_wind_df)
    
    n_days = len(response_json["days"])

    total_hours = n_days * 24

    hourly_index = np.arange(0, total_hours, 1)
    print(hourly_index)
    hourly_utc_date_time = hourly_wind_df["UTC_Time"]
    print(hourly_utc_date_time)
    hourly_date_time_df = Forecast.convert_DateTime_UTC_to_CST(hourly_utc_date_time, hourly_index)

    forecasted_wind_df = pd.merge(hourly_wind_df, hourly_date_time_df, on='UTC_Time', how='outer')
    forecasted_wind_df.drop(columns=['UTC_Time'], axis=1, inplace=True)
    forecasted_wind_df["Date_Time"] = pd.to_datetime(forecasted_wind_df["Date_Time"])
    forecasted_wind_df = forecasted_wind_df[["Date_Time", "Year", "Month", "Day", "Hour", "Temperature_F", "Humidity_percent", "WindSpeed_mph", "WindDirection_degrees", "Weather_Description"]]

    # Load the model
    scaler = load(open(r'C:\Users\admin\Desktop\austin-green-energy-predictor-master\Wind\wind_ml_model\scaler.pkl', 'rb'))
    load_nn = tf.keras.models.load_model(r'C:\Users\admin\Desktop\austin-green-energy-predictor-master\Wind\wind_ml_model\wind_model.h5', custom_objects={'mse': MeanSquaredError()})

    X = forecasted_wind_df.drop(['Date_Time', 'Weather_Description', 'Year'], axis=1)
    X_scaled = scaler.transform(X)

    y_pred = load_nn.predict(X_scaled)
    y_pred = y_pred.ravel()

    forecasted_wind_df['pred'] = y_pred

    return forecasted_wind_df.to_json(orient='records', index=False)

@app.get("/solarPredict/{year}/{month}/{day}")
def solar_predict(year: int, month: int, day: int):
    solar_day_df = solar_df.loc[(solar_df['Year'] == year) & (solar_df['Month'] == month) & (solar_df['Day'] == day)]

    scaler = load(open(r'C:\Users\admin\Desktop\austin-green-energy-predictor-master\Solar\solar_ml_model\scaler.pkl', 'rb'))
    load_nn = tf.keras.models.load_model(r'C:\Users\admin\Desktop\austin-green-energy-predictor-master\Solar\solar_ml_model\solar_model.h5', custom_objects={'mse': MeanSquaredError()})

    X = solar_day_df.drop(['Date_Time', 'Weather_Description', 'Day', 'Year', 'MWH'], axis=1)
    X_scaled = scaler.transform(X)

    y_pred = load_nn.predict(X_scaled)
    y_pred = y_pred.ravel()

    solar_day_df['pred'] = y_pred

    return solar_day_df.to_json(orient='records', index=False)

@app.get("/windPredict/{year}/{month}/{day}")
def wind_predict(year: int, month: int, day: int):
    wind_day_df = wind_df.loc[(wind_df['Year'] == year) & (wind_df['Month'] == month) & (wind_df['Day'] == day)]

    scaler = load(open(r'C:\Users\admin\Desktop\austin-green-energy-predictor-master\Wind\wind_ml_model\scaler.pkl', 'rb'))
    load_nn = tf.keras.models.load_model(r'C:\Users\admin\Desktop\austin-green-energy-predictor-master\Wind\wind_ml_model\wind_model.h5', custom_objects={'mse': MeanSquaredError()})

    X = wind_day_df.drop(['MWH', 'WindDirection_compass', 'Year', 'Weather_Description', 'Date_Time', 'WindGust_mph'], axis=1)
    X_scaled = scaler.transform(X)

    y_pred = load_nn.predict(X_scaled)
    y_pred = y_pred.ravel()

    wind_day_df['pred'] = y_pred

    return wind_day_df.to_json(orient='records', index=False)

