import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import os


file_path = "Plastic Waste Around the World.csv"

if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
    exit()

df = pd.read_csv(file_path)


df.columns = df.columns.str.strip()


country_input = input("Enter a country to forecast plastic waste growth: ").strip()


if country_input not in df['Country'].values:
    print(f"Error: Country '{country_input}' not found in the dataset. Please try again.")
    exit()


country_data = df[df["Country"] == country_input]
initial_waste = float(country_data["Total_Plastic_Waste_MT"].values[0])


years = list(range(2000, 2025))
plastic_waste = [initial_waste * (1.02 ** (year - 2024)) for year in years]  # Assuming 2% annual growth


history_df = pd.DataFrame({'Year': years, 'Plastic_Waste': plastic_waste})


model = ARIMA(history_df['Plastic_Waste'], order=(2,1,2))  
model_fit = model.fit()


forecast_years = list(range(2025, 2036))
forecast_values = model_fit.forecast(steps=11)  # Predict next 11 years


forecast_df = pd.DataFrame({'Year': forecast_years, 'Plastic_Waste': forecast_values})


fig = go.Figure()


fig.add_trace(go.Scatter(
    x=history_df['Year'], y=history_df['Plastic_Waste'],
    mode='lines+markers', name='Actual Plastic Waste',
    line=dict(color='blue')
))


fig.add_trace(go.Scatter(
    x=forecast_df['Year'], y=forecast_df['Plastic_Waste'],
    mode='lines+markers', name='Forecasted Plastic Waste',
    line=dict(color='red', dash='dot')
))


fig.update_layout(
    title=f"Forecast of {country_input}'s Plastic Waste Growth (2000-2035)",
    xaxis_title="Year",
    yaxis_title="Plastic Waste (Million Metric Tons)",
    hovermode="x",
    template="plotly_dark"
)


output_file = f"{country_input}_plastic_waste_forecast.html"
fig.write_html(output_file)

print(f"Forecast graph saved as {output_file}. Open it in a browser to view the interactive graph.")