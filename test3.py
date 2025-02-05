# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# define function to preprocess dataset
def preprocess_df(file):
    df = pd.read_csv(file, encoding='latin-1')
    
    # initially fill missing values in age as 0
    df['Age'] = df['Age'].fillna(0).astype(int)

    # change obviously wrong age to 0
    df['Age'] = df['Age'].replace(1, 0)
    df['Age'] = df['Age'].replace(152, 0)

    # convert date columns to datetime
    date_columns = ['Gift Date', 'First Gift Date', 'Last Gift Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])

    # convert amount columns to float
    amount_columns = ['Cumulative Giving',
                     'PR Commitments',
                     'SATC Commitments',
                     'FY24 Giving',
                     'FY23 Giving',
                     'FY22 Giving',
                     'FY21 Giving',
                     'FY20 Giving',
                     'FY19 Giving',
                     'FY18 Giving',
                     'Gift Amount',
                     'Fund Split Amount',
                     'Total Amount of Gifts_1',
                     'First Gift Amount',
                     'Last Gift Amount']
    for col in amount_columns:
        df[col] = df[col].fillna('').astype(str)
        df[col] = df[col].replace('[\$,]', '', regex=True).replace('^$', np.nan, regex=True).astype(float)
                
    return df

def create_figure(data, forecast, observed_name, forecast_name, observed_color, forecast_color):
    fig = go.Figure()
    
    # add observed data line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        name=observed_name,
        line=dict(color=observed_color)
    ))
    
    # add forecast line
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast.values,
        name=forecast_name,
        line=dict(color=forecast_color, dash='dash')
    ))
    
    return fig

def apply_exponential_smoothing(data):
    model = ExponentialSmoothing(data, seasonal_periods=12, trend='add', seasonal='add', damped=True)
    fit = model.fit(optimized=True)
    forecast = fit.forecast(steps=12)
    return forecast

# Set page config
st.set_page_config(page_title="Donation Analysis")

# load datasets
@st.cache_data
def load_data():
    df = preprocess_df('C:\\Users\\conduit\\Downloads\\DS Final Dataset\\df.csv')
    online = df.loc[df['Online'] == 1, ['Gift Date', 'Fund Split Amount']]
    offline = df.loc[df['Online'] == 0, ['Gift Date', 'Fund Split Amount']]
    return df, online, offline

df, online, offline = load_data()

age_bins = [0, 21, 28, 44, 60, 78, np.inf]
age_labels = ['Uncategorized',
              'Generation Z',
              'Millennials',
              'Generation X',
              'Baby Boomers',
              'Silent Generation']

df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# set the date range
date_range = pd.date_range(start='2017-07-01', end='2024-06-30', freq='M')

# resample data to monthly frequency with complete date range
online_monthly = online.set_index('Gift Date').resample('M')['Fund Split Amount'].sum()
offline_monthly = offline.set_index('Gift Date').resample('M')['Fund Split Amount'].sum()

# reindex to ensure all months are included
online_monthly = online_monthly.reindex(date_range, fill_value=0)
offline_monthly = offline_monthly.reindex(date_range, fill_value=0)

# create streamlit app
st.title('New Campaign Donation Analysis')

viz_type = st.selectbox('Select metric',
                       ['Online vs Offline Distribution'
                        , 'Gift Amount Distribution'
                        , 'Donor Type Distribution'
                        , 'Age Distribution'
                        , 'Generation Distribution'
                        , 'Giving Trends by Fiscal Year'
                        , 'Average Gift by Generation'
                        , 'Online Preference by Generation'
                        , 'Online Donation Frequency'
                        , 'Online Donations with Forecast'
                        , 'Offline Donations with Forecast'
                        , 'Total Monthly Donations with Forecast'])

if viz_type == 'Online vs Offline Distribution':
    online_counts = df['Online'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=['Offline', 'Online'], values=online_counts)])
    fig.update_traces(hoverinfo='label+value'
                      , textinfo='percent'
                      , rotation=90)
    fig.update_layout(title='Online vs Offline Distribution', template='plotly_white')
    
elif viz_type == 'Gift Amount Distribution':
    gift_amounts = df.groupby('Online')['Fund Split Amount'].sum()
    fig = go.Figure(data=[go.Pie(labels=['Offline', 'Online'], values=gift_amounts)])
    fig.update_traces(hoverinfo='label+value'
                      , textinfo='percent'
                      , rotation=90)
    fig.update_layout(title='Gift Amount Distribution', template='plotly_white')


elif viz_type == 'Donor Type Distribution':
    key_indicator_online = df.groupby(['Key Indicator', 'Online']).size().unstack()
    key_indicator_online_pct = key_indicator_online.div(key_indicator_online.sum(axis=1), axis=0)
    fig = go.Figure()
    for col in key_indicator_online_pct.columns:
        fig.add_trace(go.Bar(name='Online' if col == 1 else 'Offline'
                             , x=key_indicator_online_pct.index
                             , y=key_indicator_online_pct[col]))
    fig.update_layout(barmode='stack'
                      , title='Donor Type Distribution'
                      , xaxis_title='Key Indicator'
                      , yaxis_title='Percentage'
                      , showlegend=True)

elif viz_type == 'Age Distribution':
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[df['Age'] > 0]['Age']
                               , nbinsx=20
                               , name='Age Distribution'))
    fig.update_layout(title='Age Distribution'
                      , xaxis_title='Age'
                      , yaxis_title='Count'
                      , bargap=0.1)

elif viz_type == 'Generation Distribution':
    generation_counts = df['AgeGroup'].value_counts()
    fig = go.Figure(data=[go.Bar(x=generation_counts.index, y=generation_counts.values)])
    fig.update_layout(title='Generation Distribution'
                      , xaxis_title='Age Group'
                      , yaxis_title='Count')

elif viz_type == 'Giving Trends by Fiscal Year':
    yearly_giving = df[['FY18 Giving', 'FY19 Giving', 'FY20 Giving', 
                       'FY21 Giving', 'FY22 Giving', 'FY23 Giving', 'FY24 Giving']]
    yearly_giving = yearly_giving.rename(columns={'FY18 Giving': 'FY18', 'FY19 Giving': 'FY19'
                                                  , 'FY20 Giving': 'FY20', 'FY21 Giving': 'FY21'
                                                  , 'FY22 Giving': 'FY22', 'FY23 Giving': 'FY23'
                                                  , 'FY24 Giving': 'FY24'})
    yearly_totals = yearly_giving.sum()
    fig = go.Figure(data=[go.Bar(x=yearly_totals.index, y=yearly_totals.values)])
    fig.update_layout(title='Giving Trends by Fiscal Year'
                      , xaxis_title='Fiscal Year'
                      , yaxis_title='Total Amount')

elif viz_type == 'Average Gift by Generation':
    avg_gift_by_age = df.groupby('AgeGroup')['Fund Split Amount'].mean()
    fig = go.Figure(data=[go.Bar(x=avg_gift_by_age.index, y=avg_gift_by_age.values)])
    fig.update_layout(title='Average Gift by Generation'
                      , xaxis_title='Age Group'
                      , yaxis_title='Average Gift Amount')

elif viz_type == 'Online Preference by Generation':
    online_by_age = df.groupby('AgeGroup')['Online'].mean().sort_values(ascending=False)
    fig = go.Figure(data=[go.Bar(x=online_by_age.index, y=online_by_age.values)])
    fig.update_layout(title='Online Preference by Generation'
                      , xaxis_title='Age Group'
                      , yaxis_title='Proportion of Online Donations')

elif viz_type == 'Online Donation Frequency':
    donation_frequency = df.groupby('Constituent ID')['Online'].agg(['count', 'mean'])
    donation_frequency['frequency_bin'] = pd.cut(donation_frequency['count']
                                                 , bins=[0, 1, 2, 5, 10, 20, 50, 100, float('inf')]
                                                 , labels=['1', '2', '3-5', '6-10', '11-20', '21-50', '51-100', '100+'])
    fig = go.Figure(data=[go.Box(x=donation_frequency['frequency_bin'], y=donation_frequency['mean'])])
    fig.update_layout(title='Online Donation Frequency'
                      , xaxis_title='Number of Donations'
                      , yaxis_title='Proportion of Online Donations'
                      , xaxis=dict(categoryorder='array', categoryarray=['1', '2', '3-5', '6-10', '11-20', '21-50', '51-100', '100+']))

elif viz_type == 'Online Donations with Forecast':
    es_forecast_online = apply_exponential_smoothing(online_monthly)
    fig = create_figure(online_monthly, es_forecast_online, 'Online Observed', 'Online Forecast', 'blue', 'orange')
    fig.update_layout(title='Online Donations with Exponential Smoothing Forecast'
                      , xaxis_title='Date'
                      , yaxis_title='Gift Amount ($)'
                      , hovermode='x unified'
                      , showlegend=True
                      , template='plotly_white'
                      , xaxis=dict(rangeslider=dict(visible=True), type='date'))    

elif viz_type == 'Offline Donations with Forecast':
    es_forecast_offline = apply_exponential_smoothing(offline_monthly)
    fig = create_figure(offline_monthly, es_forecast_offline, 'Offline Observed', 'Offline Forecast', 'red', 'orange')
    fig.update_layout(title='Offline Donations with Exponential Smoothing Forecast'
                      , xaxis_title='Date'
                      , yaxis_title='Gift Amount ($)'
                      , hovermode='x unified'
                      , showlegend=True
                      , template='plotly_white'
                      , xaxis=dict(rangeslider=dict(visible=True), type='date'))

else:
    total_monthly = online_monthly + offline_monthly
    es_forecast_total = apply_exponential_smoothing(total_monthly)
    fig = create_figure(total_monthly, es_forecast_total, 'Total Observed', 'Total Forecast', 'purple', 'orange')
    fig.update_layout(title='Total Monthly Donations with Exponential Smoothing Forecast'
                      , xaxis_title='Date'
                      , yaxis_title='Gift Amount ($)'
                      , hovermode='x unified'
                      , showlegend=True
                      , template='plotly_white'
                      , xaxis=dict(rangeslider=dict(visible=True), type='date'))

st.plotly_chart(fig, use_container_width=True)