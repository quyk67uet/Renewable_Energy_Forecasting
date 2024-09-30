# Import dependencies
import pandas as pd
import datetime
from dateutil import tz
import requests
import numpy as np

def convert_DateTime_UTC_to_CST(UTC_datetime_list, list_range):
    CST_datetime_list = []

    for date in list_range:    
        # Convert the date/time to ISO standard in string format
        date_time = datetime.datetime.utcfromtimestamp(UTC_datetime_list[date]).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create a datetime object, representing the UTC time
        time_utc = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')

        # Replace the timezone field of the datetime object to UTC
        from_zone = tz.gettz('UTC')
        time_utc = time_utc.replace(tzinfo=from_zone)

        # Convert time zone from UTC to central
        to_zone = tz.gettz('America/Chicago')
        time_cst = time_utc.astimezone(to_zone)

        # Append the date/time, year, month, day, and hour
        CST_datetime_list.append({
            "UTC_Time": UTC_datetime_list[date],
            "Date_Time": time_cst.strftime('%Y-%m-%d %H:%M:%S'),
            "Year": time_cst.year,
            "Month":time_cst.month,
            "Day":time_cst.day,
            "Hour":time_cst.hour
            })

    datetimeDataFrame = pd.DataFrame(CST_datetime_list)
    
    return datetimeDataFrame


def calculate_sunhour(sunrise_list, sunset_list, list_range):
   
    sunhour_list = []

    for day in list_range:
        # Convert the date/time to ISO standard in string format
        sunrise_date_time = datetime.datetime.utcfromtimestamp(sunrise_list[day]).strftime('%Y-%m-%d %H:%M:%S')
        sunset_date_time = datetime.datetime.utcfromtimestamp(sunset_list[day]).strftime('%Y-%m-%d %H:%M:%S')

        # Create a datetime object, representing the UTC time
        sunrise_utc = datetime.datetime.strptime(sunrise_date_time, '%Y-%m-%d %H:%M:%S')
        sunset_utc = datetime.datetime.strptime(sunset_date_time, '%Y-%m-%d %H:%M:%S')

        # Replace the timezone field of the datetime object to UTC
        from_zone = tz.gettz('UTC')
        
        sunrise_utc = sunrise_utc.replace(tzinfo=from_zone)
        sunset_utc = sunset_utc.replace(tzinfo=from_zone)

        # Convert time zone from UTC to central
        to_zone = tz.gettz('America/Chicago')
        
        sunrise_cst = sunrise_utc.astimezone(to_zone)
        sunset_cst = sunset_utc.astimezone(to_zone)
        
        # Convert to string
        sunrise_str = sunrise_cst.strftime('%Y-%m-%d %H:%M:%S')
        sunset_str = sunset_cst.strftime('%Y-%m-%d %H:%M:%S')

        # Calculate Sunhour
        sunrise = datetime.datetime.strptime(sunrise_str, '%Y-%m-%d %H:%M:%S')
        sunset = datetime.datetime.strptime(sunset_str, '%Y-%m-%d %H:%M:%S')
        Sunhour_timedelta = sunset - sunrise
        Sunhour_seconds = Sunhour_timedelta.seconds
        Sunhour = Sunhour_seconds / 3600

        # Append to List
        sunhour_list.append({
            "Sunrise": sunrise_list[day],
            "Sunhour": Sunhour
        })
        
    sunhourDataFrame = pd.DataFrame(sunhour_list)
    
    return sunhourDataFrame

def makeAPIRequest(lat, lon, weather_api_key):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}"
    params = {
        'key': weather_api_key,
        'include': 'current,hours,days',
        'unitGroup': 'metric'
    }
    response = requests.get(url, params=params)
    return response.json()

def current_solar_weather(responseJson):
    # Convert the json response to a pandas dataframe
    current_weather_DF = pd.DataFrame([{
        "UTC_Time": responseJson["currentConditions"]["datetimeEpoch"],
        "Temperature_F": responseJson["currentConditions"]["temp"], 
        "Humidity_percent": responseJson["currentConditions"]["humidity"],
        "CloudCover_percent": responseJson["currentConditions"]["cloudcover"],
        "uvIndex": responseJson["currentConditions"]["uvindex"],
        "Sunrise": responseJson["currentConditions"]["sunrise"],
        "Weather_Description": responseJson["currentConditions"]["conditions"]
    }])

    return current_weather_DF

def forecasted_daily_solar(responseJson):
    # Initiate list
    forecasted_daily_weather = []

    # Append json response to list
    for day in responseJson["days"]:
        try:
            forecasted_daily_weather.append({
                "UTC_Time": day["datetimeEpoch"],  
                "Sunrise": day["sunriseEpoch"],    
                "Sunset": day["sunsetEpoch"],       
                "uvIndex": day.get("uvindex", 1)    
            })
        except KeyError:
            forecasted_daily_weather.append({
                "UTC_Time": day["datetimeEpoch"],  
                "Sunrise": day["sunriseEpoch"],    
                "Sunset": day["sunsetEpoch"],       
                "uvIndex": 1                         
            })


    # Convert list to pandas dataframe
    daily_weather_DF = pd.DataFrame(forecasted_daily_weather)
    
    return daily_weather_DF

def forecasted_hourly_solar(responseJson):
    # Initiate list
    forecasted_hourly_weather = []

    # Append json response to list
    for day in responseJson["days"]:
        for hour in day["hours"]:
            forecasted_hourly_weather.append({
                "UTC_Time": hour["datetimeEpoch"],
                "Temperature_F": hour["temp"],
                "Weather_Description": hour["conditions"],
                "CloudCover_percent": hour["cloudcover"],
                "Humidity_percent": hour["humidity"]
            })

    # Convert list to pandas dataframe
    hourly_weather_DF = pd.DataFrame(forecasted_hourly_weather)
    
    return hourly_weather_DF

def current_wind_weather(responseJson):
    # Convert the json response to a pandas dataframe
    current_weather_DF = pd.DataFrame([{
        "UTC_Time": responseJson["currentConditions"]["datetimeEpoch"],
        "Temperature_F": responseJson["currentConditions"]["temp"], 
        "Weather_Description": responseJson["currentConditions"]["conditions"],
        "Humidity_percent": responseJson["currentConditions"]["humidity"],
        "WindSpeed_mph": responseJson["currentConditions"]["windspeed"],
        "WindDirection_degrees": responseJson["currentConditions"]["winddir"]
    }])

    return current_weather_DF

def forecasted_hourly_wind(responseJson):
    # Initiate list
    forecasted_hourly_weather = []

    # Append json response to list
    for day in responseJson["days"]:
        for hour in day["hours"]:
            forecasted_hourly_weather.append({
                "UTC_Time": hour["datetimeEpoch"],  
                "Temperature_F": hour["temp"],       
                "Weather_Description": hour["conditions"], 
                "Humidity_percent": hour["humidity"], 
                "WindSpeed_mph": hour["windspeed"],   
                "WindDirection_degrees": hour["winddir"]  
            })

    # Convert list to pandas dataframe
    hourly_weather_DF = pd.DataFrame(forecasted_hourly_weather)
    
    return hourly_weather_DF


def modelPrediction(forecasted_weather_df, X_scaled, load_nn):  
    # Predict values for test set
    y_pred = load_nn.predict(X_scaled)
    y_pred = y_pred.ravel()

    # Create dataframe for results
    nn_results = pd.DataFrame()
    nn_results['pred'] = y_pred
    nn_results['Hour'] = forecasted_weather_df['Hour']
    nn_results['Day'] = forecasted_weather_df['Day']
    nn_results['Date_Time'] = forecasted_weather_df['Date_Time']
    
    return nn_results