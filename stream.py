import streamlit as st
import requests
import pandas as pd
from io import StringIO
import json
import plotly.express as px
import matplotlib.pyplot as plt
import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv 

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_KEY")

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="Renewable energy forecasting ☀️🌬️", page_icon="⚡", layout="wide")

st.sidebar.header("Chọn loại dự đoán")
option = st.sidebar.selectbox("Chọn một tùy chọn:", ["Dự đoán Năng lượng Mặt Trời ☀️", "Dự đoán Năng lượng Gió 🌬️", "Dữ liệu Năng lượng Mặt Trời ☀️", "Dữ liệu Năng lượng Gió 🌬️"])

# Function to reset chat history and energy content
def reset_chat_and_content():
    st.session_state.chat_history = []
    st.session_state.energy_content = ""

# Function to generate chatbot response using Google Gemini
def generate_response(energy_content, user_prompt):
    prompt = (
        f"Bạn là một chuyên gia dự báo năng lượng. Dưới đây là dữ liệu dự báo về năng lượng mặt trời và gió:\n\n"
        f"{energy_content}\n\n"
        f"Câu hỏi của người dùng: {user_prompt}\n\n"
        f"Hãy cung cấp câu trả lời phù hợp và chính xác dựa trên dữ liệu dự báo năng lượng:"
    )
    response = model.generate_content(prompt)
    return response.text

# Initialize session state variables if not present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'energy_content' not in st.session_state:
    st.session_state.energy_content = ""

# Reset energy content if the task changes
if option != st.session_state.get('last_task'):
    reset_chat_and_content()
st.session_state.last_task = option

def generate_response(energy_content, user_prompt):
        prompt = (
            f"Bạn là một chuyên gia dự báo năng lượng. Dưới đây là dữ liệu dự báo về năng lượng mặt trời và gió:\n\n"
            f"{energy_content}\n\n"
            f"Câu hỏi của người dùng: {user_prompt}\n\n"
            f"Hãy cung cấp câu trả lời phù hợp và chính xác dựa trên dữ liệu dự báo năng lượng:"
        )
        response = model.generate_content(prompt)
        return response.text

if option == "Dự đoán Năng lượng Mặt Trời ☀️":
    st.title("Dự đoán Năng lượng Mặt Trời ☀️")
    st.sidebar.subheader("Chọn ngày dự đoán")
    year = int(st.sidebar.number_input("Năm", min_value=2000, max_value=2100, value=2024))
    month = int(st.sidebar.number_input("Tháng", min_value=1, max_value=12, value=9))
    day = int(st.sidebar.number_input("Ngày", min_value=1, max_value=31, value=28))

    # Combine the year, month, and day into a date object
    selected_date = datetime.date(year, month, day)

    if st.sidebar.button("Dự đoán"):
        response = requests.get(f"http://127.0.0.1:8000/solarPredict/{year}/{month}/{day}")
        if response.status_code == 200:
            energy_content = json.loads(response.json())
            df = pd.DataFrame(energy_content)
            st.write(df)
            
            # Convert Date_Time column to datetime for plotting
            df['Date_Time'] = pd.to_datetime(df['Date_Time'])
            
            # Line plot: Predicted Solar Energy Production over time
            plt.figure(figsize=(12, 6))
            plt.plot(df['Date_Time'], df['pred'], marker='o', label='Dự đoán (pred)', color='blue')
            plt.title(f'Dự Đoán Sản Lượng Điện Ngày {selected_date}')
            plt.xlabel('Thời Gian')
            plt.ylabel('Sản Lượng Điện (pred)')
            plt.xticks(rotation=45)
            plt.grid()
            plt.legend()
            st.pyplot(plt)
            
            # Line chart using Plotly for Actual vs Predicted MWH
            fig = px.line(df, x='Date_Time', y=['MWH', 'pred'],
                        labels={'value': 'Megawatt Hours (MWH)', 'variable': 'Type'},
                        title=f"Actual vs Predicted Solar Energy Production - {selected_date}")
            fig.update_layout(legend_title_text='')
            st.plotly_chart(fig)

            # Bar chart: Temperature and Humidity
            fig2 = px.bar(df, x='Date_Time', y=['Temperature_F', 'Humidity_percent'],
                        labels={'value': 'Value', 'variable': 'Metric'},
                        title=f"Temperature and Humidity - {selected_date}")
            fig2.update_layout(legend_title_text='')
            st.plotly_chart(fig2)

            # Scatter plot: Cloud Cover vs Solar Energy Production
            fig3 = px.scatter(df, x='CloudCover_percent', y='MWH',
                            color='Hour',  # Color points by the hour
                            labels={'CloudCover_percent': 'Cloud Cover (%)', 'MWH': 'Megawatt Hours (MWH)'},
                            title=f"Cloud Cover vs Solar Energy Production - {selected_date}")
            st.plotly_chart(fig3)

            # New chart: UV Index and Sunhour by Hour
            fig4 = px.line(df, x='Hour', y=['uvIndex', 'Sunhour'],
                        labels={'value': 'Value', 'variable': 'Metric'},
                        title=f"UV Index and Sunhour by Hour - {selected_date}")
            fig4.update_layout(legend_title_text='')
            st.plotly_chart(fig4)

            # Displaying summary information
            st.subheader("Daily Summary")
            total_actual_mwh = df['MWH'].sum()
            total_predicted_mwh = df['pred'].sum()
            accuracy = (1 - abs(total_actual_mwh - total_predicted_mwh) / total_actual_mwh) * 100
            st.write(f"Total Actual MWH: {total_actual_mwh:.2f}")
            st.write(f"Total Predicted MWH: {total_predicted_mwh:.2f}")
            st.write(f"Prediction Accuracy: {accuracy:.2f}%")

            # Weather conditions summary
            st.subheader("Weather Conditions")
            weather_counts = df['Weather_Description'].value_counts()
            st.write("Weather conditions throughout the day:")
            st.write(weather_counts)

            # Update energy content for chatbot
            st.session_state.energy_content = "\n\n".join(df.apply(lambda x: f"{x['Date_Time']}: {x['pred']} MW", axis=1)) 
        else:
            st.error("Lỗi trong việc lấy dữ liệu.")

elif option == "Dự đoán Năng lượng Gió 🌬️":
    st.title("Dự đoán Năng lượng Gió 🌬️")
    st.sidebar.subheader("Chọn ngày dự đoán")
    year = int(st.sidebar.number_input("Năm", min_value=2000, max_value=2100, value=2024))
    month = int(st.sidebar.number_input("Tháng", min_value=1, max_value=12, value=9))
    day = int(st.sidebar.number_input("Ngày", min_value=1, max_value=31, value=28))

    # Combine the year, month, and day into a date object
    selected_date = datetime.date(year, month, day)
    if st.sidebar.button("Dự đoán"):
        response = requests.get(f"http://127.0.0.1:8000/windPredict/{year}/{month}/{day}")
        if response.status_code == 200:
            energy_content = json.loads(response.json())
            df = pd.DataFrame(energy_content)
            st.write(df)
            
            # Convert Date_Time column to datetime for plotting
            df['Date_Time'] = pd.to_datetime(df['Date_Time'])

            # Line plot for predicted energy production
            plt.figure(figsize=(12, 6))
            plt.plot(df['Date_Time'], df['pred'], marker='o', label='Dự đoán (pred)', color='blue')
            plt.title(f'Dự Đoán Sản Lượng Điện Ngày {selected_date}')
            plt.xlabel('Thời Gian')
            plt.ylabel('Sản Lượng Điện (pred)')
            plt.xticks(rotation=45)
            plt.grid()
            plt.legend()
            st.pyplot(plt)

            # Line chart using Plotly for actual vs predicted MWH
            fig = px.line(df, x='Date_Time', y=['MWH', 'pred'],
                            labels={'value': 'Megawatt Hours (MWH)', 'variable': 'Type'},
                            title=f"Actual vs Predicted Wind Energy Production - {selected_date}")
            fig.update_layout(legend_title_text='')
            st.plotly_chart(fig)

            # Bar chart for temperature and humidity
            fig2 = px.bar(df, x='Date_Time', y=['Temperature_F', 'Humidity_percent'],
                            labels={'value': 'Value', 'variable': 'Metric'},
                            title=f"Temperature and Humidity - {selected_date}")
            fig2.update_layout(legend_title_text='')
            st.plotly_chart(fig2)

            # Scatter plot: Wind Speed vs Wind Energy Production
            fig3 = px.scatter(df, x='WindSpeed_mph', y='MWH',
                                color='Hour',  # Color points by the hour
                                labels={'WindSpeed_mph': 'Wind Speed (mph)', 'MWH': 'Megawatt Hours (MWH)'},
                                title=f"Wind Speed vs Wind Energy Production - {selected_date}")
            st.plotly_chart(fig3)

            # Scatter plot: Wind Gust vs Wind Energy Production
            fig4 = px.scatter(df, x='WindGust_mph', y='MWH',
                                color='Hour',  # Color points by the hour
                                labels={'WindGust_mph': 'Wind Gust (mph)', 'MWH': 'Megawatt Hours (MWH)'},
                                title=f"Wind Gust vs Wind Energy Production - {selected_date}")
            st.plotly_chart(fig4)

            # Scatter plot: Wind Direction vs Wind Energy Production
            fig5 = px.scatter(df, x='WindDirection_degrees', y='MWH',
                                color='Hour',  # Color points by the hour
                                labels={'WindDirection_degrees': 'Wind Direction (degrees)', 'MWH': 'Megawatt Hours (MWH)'},
                                title=f"Wind Direction vs Wind Energy Production - {selected_date}")
            st.plotly_chart(fig5)

            # Displaying daily summary
            st.subheader("Daily Summary")
            total_actual_mwh = df['MWH'].sum()
            total_predicted_mwh = df['pred'].sum()
            accuracy = (1 - abs(total_actual_mwh - total_predicted_mwh) / total_actual_mwh) * 100
            st.write(f"Total Actual MWH: {total_actual_mwh:.2f}")
            st.write(f"Total Predicted MWH: {total_predicted_mwh:.2f}")
            st.write(f"Prediction Accuracy: {accuracy:.2f}%")

            # Weather conditions summary
            st.subheader("Weather Conditions")
            weather_counts = df['Weather_Description'].value_counts()
            st.write("Weather conditions throughout the day:")
            st.write(weather_counts)

            # Update energy content for chatbot
            st.session_state.energy_content = "\n\n".join(df.apply(lambda x: f"{x['Date_Time']}: {x['pred']} MW", axis=1))

        else:
            st.error("Lỗi trong việc lấy dữ liệu.")

elif option == "Dữ liệu Năng lượng Mặt Trời ☀️":
    if st.sidebar.button("Lấy dữ liệu"):
        response = requests.get("http://127.0.0.1:8000/getSolar")
        if response.status_code == 200:
            energy_content = json.loads(response.json())
            df = pd.DataFrame(energy_content)
            st.write(df)

            st.session_state.energy_content = "\n\n".join(df.apply(lambda x: f"{x['Date_Time']}: {x['pred']} MW", axis=1))
        else:
            st.error("Lỗi trong việc lấy dữ liệu.")

elif option == "Dữ liệu Năng lượng Gió 🌬️":
    if st.sidebar.button("Lấy dữ liệu"):
        response = requests.get("http://127.0.0.1:8000/getwind")
        if response.status_code == 200:
            energy_content = json.loads(response.json())
            df = pd.DataFrame(energy_content)
            st.write(df)

            st.session_state.energy_content = "\n\n".join(df.apply(lambda x: f"{x['Date_Time']}: {x['pred']} MW", axis=1))
        else:
            st.error("Lỗi trong việc lấy dữ liệu.")

# Chatbot functionality
if st.session_state.energy_content:
    st.subheader("Chat based on Energy Content")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask about the energy content...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        bot_response = generate_response(st.session_state.energy_content, user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

        with st.chat_message("assistant"):
            st.markdown(bot_response)
else:
    st.info("Please make a prediction to start chatting based on the energy content.")