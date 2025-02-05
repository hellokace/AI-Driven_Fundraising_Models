# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# create function to load and preprocess data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding='latin-1', usecols=['Gift Date', 'Fund Split Amount'])
        data['Gift Date'] = pd.to_datetime(data['Gift Date'])
        data['Fund Split Amount'] = data['Fund Split Amount'].replace('[\$,]', '', regex=True).replace('^$', np.nan, regex=True).astype(float)
        return data.sort_values('Gift Date')
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {str(e)}")
        return None

# load datasets
online = load_data('C:\\Users\\conduit\\Downloads\\DS Final Dataset\\online_donations.csv')
offline = load_data('C:\\Users\\conduit\\Downloads\\DS Final Dataset\\offline_donations.csv')

# create visualization for forecasting online and offline donations
if online is not None and offline is not None:
    online['Online'] = 1
    offline['Online'] = 0

    # set the date range
    date_range = pd.date_range(start='2017-07-01', end='2024-06-30', freq='M')

    # resample data to monthly frequency with complete date range
    online_monthly = online.set_index('Gift Date').resample('M')['Fund Split Amount'].sum()
    offline_monthly = offline.set_index('Gift Date').resample('M')['Fund Split Amount'].sum()

    # reindex to ensure all months are included
    online_monthly = online_monthly.reindex(date_range, fill_value=0)
    offline_monthly = offline_monthly.reindex(date_range, fill_value=0)

    # create streamlit app
    st.title('Monthly Donation Analysis')

    viz_type = st.selectbox('Select visualization type',
                            ['Online Donations with Forecast',
                             'Offline Donations with Forecast',
                             'Total Monthly Donations'])

    def create_figure(data, forecast, observed_name, forecast_name, observed_color, forecast_color):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=data.index
                                 ,y=data.values
                                 ,name=observed_name
                                 ,line=dict(color=observed_color)))
        
        fig.add_trace(go.Scatter(x=forecast.index
                                 ,y=forecast.values
                                 ,name=forecast_name
                                 ,line=dict(color=forecast_color, dash='dash')))
        
        return fig

    def apply_exponential_smoothing(data):
        model = ExponentialSmoothing(data, seasonal_periods=12, trend='add', seasonal='add', damped=True)
        fit = model.fit(optimized=True)
        forecast = fit.forecast(steps=12)
        return forecast

    if viz_type == 'Online Donations with Forecast':
        es_forecast_online = apply_exponential_smoothing(online_monthly)
        fig = create_figure(online_monthly, es_forecast_online, 'Online Observed', 'Online Forecast', 'blue', 'orange')
        title = 'Online Donations with Exponential Smoothing Forecast'

    elif viz_type == 'Offline Donations with Forecast':
        es_forecast_offline = apply_exponential_smoothing(offline_monthly)
        fig = create_figure(offline_monthly, es_forecast_offline, 'Offline Observed', 'Offline Forecast', 'red', 'orange')
        title = 'Offline Donations with Exponential Smoothing Forecast'

    else:
        total_monthly = online_monthly + offline_monthly
        es_forecast_total = apply_exponential_smoothing(total_monthly)
        fig = create_figure(total_monthly, es_forecast_total, 'Total Observed', 'Total Forecast', 'purple', 'orange')
        title = 'Total Monthly Donations with Exponential Smoothing Forecast'

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Gift Amount ($)',
        hovermode='x unified',
        showlegend=True,
        template='plotly_white',
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        )
    )

    st.plotly_chart(fig)
else:
    st.error("Failed to load data. Please check your file paths and try again.")

### new ones

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('C:\\Users\\conduit\\Downloads\\DS Final Dataset\\df.csv', encoding='latin-1')
    return df

df = load_data()

# Create two columns for the pie charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Donation Method Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    online_counts = df['Online'].value_counts()
    plt.pie(online_counts, labels=['Offline', 'Online'], autopct='%1.1f%%')
    plt.title('Proportion of Online vs Offline Donations')
    st.pyplot(fig1)

with col2:
    st.subheader("Gift Amount Distribution")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    gift_amounts = df.groupby('Online')['Fund Split Amount'].sum()
    plt.pie(gift_amounts, labels=['Offline', 'Online'], autopct='%1.1f%%')
    plt.title('Proportion of Gift Amount: Online vs Offline')
    st.pyplot(fig2)

# Add metrics below the charts
st.subheader("Key Metrics")
metrics_col1, metrics_col2 = st.columns(2)

with metrics_col1:
    total_online = df[df['Online'] == 1]['Fund Split Amount'].sum()
    st.metric("Total Online Donations", f"${total_online:,.2f}")

with metrics_col2:
    total_offline = df[df['Online'] == 0]['Fund Split Amount'].sum()
    st.metric("Total Offline Donations", f"${total_offline:,.2f}")