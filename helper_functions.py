import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.signal import savgol_filter
import json
from datetime import datetime
import statsmodels.api as sm
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from helper_functions import *

import requests
import time
from causalimpact import CausalImpact

from tqdm.notebook import tqdm
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import threading
import csv
import gc
from lxml import etree
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from ipywidgets import interact, VBox
from IPython.display import display, clear_output
import plotly.io as pio
from matplotlib.lines import Line2D

import dash
from dash import dcc, html

from scipy.stats import pearsonr



  # Output html that you can copy paste
#  fig.to_html(full_html=False, include_plotlyjs='cdn')
  # Saves a html doc that you can copy paste
#  fig.write_html("output.html", full_html=False, include_plotlyjs='cdn')



def load_interventions():
    interventions = pd.read_csv('interventions.csv')
    interventions.set_index('lang', inplace=True)
    return interventions

def load_applemob():
    applemob = pd.read_csv('applemobilitytrends-2020-04-20.csv')
    return applemob

def load_globalmob():
    globalmob = pd.read_csv('Global_Mobility_Report.csv')
    return globalmob

def load_aggregated_timeseries():
    with open('aggregated_timeseries.json', 'r') as f:
        # Load the JSON data
        d = json.load(f)
    return d

def choose_restrictiveness(choice, english):
    if choice == "All":
        data = {
            'France': ['fr', 'FR'],
            'Denmark': ['da', 'DK'],
            'Germany': ['de', 'DE'],
            'Italy': ['it', 'IT'],
            'Netherlands': ['nl', 'NL'],
            'Norway': ['no', 'NO'],
            'Serbia': ['sr', 'RS'],
            'Sweden': ['sv', 'SE'],
            'Korea': ['ko', 'KR'],
            'Catalonia': ['ca', 'ES'],
            'Finland': ['fi', 'FI'],
            'Japan': ['ja', 'JP'],
            }
    if choice == "Restrictive":
        data = {
            'France': ['fr', 'FR'],
            'Italy': ['it', 'IT'],
            'Serbia': ['sr', 'RS'],
            'Catalonia': ['ca', 'ES'],
            }
    if choice == "Semi-Restrictive":
        data = {
            'Denmark': ['da', 'DK'],
            'Germany': ['de', 'DE'],
            'Netherlands': ['nl', 'NL'],
            'Norway': ['no', 'NO'],
            'Finland': ['fi', 'FI'],
            }
    if choice == "Unrestrictive":
        data = {
            'Sweden': ['sv', 'SE'],
            'Korea': ['ko', 'KR'],
            'Japan': ['ja', 'JP'],
            }
    if english == "Yes":
        data['England'] = ['en', 'GB']

    df_code = pd.DataFrame(data)
    df_code = df_code.transpose()
    df_code.rename(columns = {0:'lang', 1:'state'}, inplace = True)
    return data, df_code

def average_mobility(d, df_code, interventions, globalmob):
    # Create subplot
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Mobility'], vertical_spacing=0.1)

    # Initialize a list to store individual mobility lines
    all_lines = []

    for i, c in enumerate(df_code['lang']):
        cs = df_code.iloc[i]['state']

        if cs == 'KR':
            globalmob_ko = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull()) & (globalmob['metro_area'].isnull())]
        else:
            if cs == 'RS':
                globalmob_sr = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['metro_area'].isnull())]
            else:
                if cs == 'ES':
                    globalmob_ca = globalmob[(globalmob['country_region_code'] == 'ES') & (globalmob['sub_region_1'] == 'Catalonia') & (globalmob['sub_region_2'].isnull())].copy()
                else:
                    globalmob_g = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull())].copy()
                    globalmob_g.reset_index(inplace=True, drop=True)

        df = globalmob_g.copy(deep=True)

        mobility_g = interventions.loc[c]['Mobility']
        lockdown_g = interventions.loc[c]['Lockdown']
        normalcy_g = interventions.loc[c]['Normalcy']

        columns = globalmob.columns[8:]
        df = df.drop(['residential_percent_change_from_baseline', 'parks_percent_change_from_baseline'], axis=1)
        columns = columns.drop(['residential_percent_change_from_baseline', 'parks_percent_change_from_baseline'])

        mean_g = df[columns].mean(axis=1)

        # Light grey lines for individual countries
        for column in columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df[column], mode='lines', line=dict(color='lightgrey', width=1.5), showlegend=False, opacity=0.25))
        
        # Vertical lines
        if cs =='GB':
            fig.add_trace(go.Scatter(x=[lockdown_g, lockdown_g], y=[-100, 100], mode='lines', line=dict(color='black', width=1.5), name=f'Lockdown', showlegend=True))
            fig.add_trace(go.Scatter(x=[normalcy_g, normalcy_g], y=[-100, 100], mode='lines', line=dict(color='black', width=1.5, dash='dash'), name=f'Normalcy', showlegend=True))
        else:
            fig.add_trace(go.Scatter(x=[lockdown_g, lockdown_g], y=[-100, 100], mode='lines', line=dict(color='black', width=1.5), name=f'Lockdown {c}', showlegend=False))
            fig.add_trace(go.Scatter(x=[normalcy_g, normalcy_g], y=[-100, 100], mode='lines', line=dict(color='black', width=1.5, dash='dash'), name=f'Normalcy {c}', showlegend=False))

        # Plot individual lines
        if cs == 'GB':
            line_label = f'Average Mobility in {df_code.index[i]}'
            line_color = 'red'
            line_width = 4
            fig.add_trace(go.Scatter(x=df['date'], y=mean_g, mode='lines', name=line_label, line=dict(color=line_color, width=line_width), showlegend=True))
        else:
            line_label = '_nolegend_'
            line_color = 'grey'
            line_width = 1.5
            fig.add_trace(go.Scatter(x=df['date'], y=mean_g, mode='lines', name=line_label, line=dict(color=line_color, width=line_width), showlegend=False, opacity=0.5))

        # Add individual lines to the list
        all_lines.append(mean_g)

    # Calculate the average line for all countries
    average_line = np.mean(all_lines, axis=0)

    # Plot the average line as a thick blue line
    fig.add_trace(go.Scatter(x=df['date'], y=average_line, mode='lines', name='Average Mobility (All Countries)', line=dict(color='blue', width=4)))

    # Customize layout
    fig.update_layout(
        title='Comparing Normalized Percentage of Wikipedia page views related to video games to English',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Percentage of Mobility Compared to Day 0'),
        legend=dict(x=1.02, y=1),
        showlegend=True,
        height=600,
        width=900,
    )

    fig.show()
    return

def plot_percent_pageviews(d, df_code, interventions):
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Percentage of Wikipedia page views related to video games'])
    all_lines = []
    max_length = 0  # Track the maximum length of y_fit arrays

    for i, c in enumerate(df_code['lang']):
        dt = d[c]["topics"]["Culture.Media.Video games"]["percent"]

        mobility_g = interventions.loc[c]['Mobility']
        format_string = "%Y-%m-%d"

        # Convert the string to a numpy.datetime64 object
        date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

        dates = list(dt.keys())
        numbers = list(dt.values())

        dates = pd.to_datetime(dates)

        if c == 'sv':
            x = [datetime.timestamp(k) for k in dates]
            x = x[365:]
            y = [val for val in numbers if not math.isnan(val)]
        else:
            x = [datetime.timestamp(k) for k in dates]
            y = numbers

        degree = 4
        coefficients = np.polyfit(x, y, degree)
        polynomial = np.poly1d(coefficients)

        y_fit = polynomial(x)

        # Track the maximum length
        max_length = max(max_length, len(y_fit))

        # Plot individual lines
        fig.add_trace(go.Scatter(x=dates, y=numbers, mode='lines', line=dict(color='lightgrey', width=0.5), showlegend=False, opacity=0.5))
        
        if c =='fr':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='green', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='da':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='orange', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='de':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='yellow', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='it':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='purple', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='nl':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='pink', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='no':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='black', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='sr':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='grey', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='sv':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='brown', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='ko':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='mediumorchid', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='ca':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='tan', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='fi':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='olive', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='ja':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='dodgerblue', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))
        if c =='en':
            fig.add_trace(go.Scatter(x=dates, y=y_fit, mode='lines', line=dict(color='lawngreen', width=1.5), name=f'{df_code.index[i]} - Trend Line', showlegend=True))

        fig.add_shape(go.layout.Shape(
            type='line',
            x0=date_object,
            x1=date_object,
            y0=0,
            y1=1,
            line=dict(color='blue', width=1.5, dash='dash'),
            layer='below'
        ))

        # Add individual lines to the list
        all_lines.append(y_fit)

    # Pad shorter arrays with NaN values
    all_lines_padded = [np.pad(line, (0, max_length - len(line)), 'constant', constant_values=np.nan) for line in all_lines]

    # Calculate the average line for all countries
    average_line = np.nanmean(all_lines_padded, axis=0)

    # Plot the average line as a thick blue line
    fig.add_trace(go.Scatter(x=dates, y=average_line, mode='lines', name='Average Trend', line=dict(color='red', width=3)))

    # Update layout
    fig.update_layout(
        xaxis=dict(title='Date', tickangle=45, tickmode='array'),
        yaxis=dict(title='Percentage', range=[0, 0.015]),
        showlegend=True,
        height=600,
        width=800,
    )

    fig.show()
    return

def plot_normalized_percent_pageviews(d, df_code, interventions):
# Assuming df_code, d, and interventions are defined as in your Matplotlib code

    fig = go.Figure()

    all_lines = []
    max_length = 0  # Track the maximum length of y_fit arrays

    for i, c in enumerate(df_code['lang']):
        dt = d[c]["topics"]["Culture.Media.Video games"]["percent"]

        mobility_g = interventions.loc[c]['Mobility']
        format_string = "%Y-%m-%d"

        # Convert the string to a numpy.datetime64 object
        date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

        dates = list(dt.keys())
        numbers = list(dt.values())

        dates = pd.to_datetime(dates)

        if c == 'en':
            x = [datetime.timestamp(k) for k in dates]
            y = [val for val in numbers if not math.isnan(val)]
            line_color = 'red'
            line_width = 6  # Set thickness for England line
            x2 = [datetime.timestamp(k) for k in dates]
            y2 = [val for val in numbers if not math.isnan(val)]

        else:
            x = [datetime.timestamp(k) for k in dates]
            y = numbers
            line_color = 'grey'
            line_width = 2  # Set default thickness for other lines

        degree = 4
        coefficients = np.polyfit(x, y, degree)
        polynomial = np.poly1d(coefficients)

        y_fit = polynomial(x)

        index = dates.get_loc(date_object)
        mean = y_fit[0:index].mean()
        offset = 0 - mean
        y_fit = y_fit + offset

        if c == 'en':
            fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=y_fit,
                                mode='lines',
                                name=f'{df_code.index[i]} - Trend Line',
                                line=dict(color=line_color, width=line_width),
                                showlegend=True))
            coefficients2 = np.polyfit(x2, y2, degree)
            polynomial2 = np.poly1d(coefficients2)
            index2 = dates.get_loc(date_object)
            mean2 = y_fit[0:index].mean()
            offset2 = 0 - mean2
            y_fit2 = polynomial(x2)
            y_fit2 = y_fit2 + offset
            y_fit2 = y_fit2 / 2.5
            max_length = max(max_length, len(y_fit2))
            fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=y_fit2,
                                    mode='lines',
                                    name=f'Scaled England - Trend Line',
                                    line=dict(color='orange', width=line_width)))

        max_length = max(max_length, len(y_fit))
        
        if c != 'en':
            fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=y_fit,
                                    mode='lines',
                                    name=f'{df_code.index[i]} - Trend Line',
                                    line=dict(color=line_color, width=line_width),
                                    showlegend=False))

        all_lines.append(y_fit)

    # Pad shorter arrays with NaN values
    all_lines_padded = all_lines
    all_lines_padded = np.array(all_lines_padded)
    # Calculate the average line for all countries
    average_line = np.nanmean(all_lines_padded, axis=0)
    average_line_restrictive = np.nanmean(all_lines_padded[[0,3,6,9,12],:], axis=0)
    average_line_semi = np.nanmean(all_lines_padded[[1,2,4,5,10,12],:], axis=0)
    average_line_unrestrictive = np.nanmean(all_lines_padded[[7,8,11,12],:], axis=0)

    # Plot the average line as a thick blue line
    fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=average_line,
                            mode='lines',
                            name='Average Trend',
                            line=dict(color='blue', width=6)))
    fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=average_line_restrictive,
                            mode='lines',
                            name='Average Restrictive Trend',
                            line=dict(color='blue', width=6)))
    fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=average_line_semi,
                            mode='lines',
                            name='Average Semi-Restrictive Trend',
                            line=dict(color='blue', width=6)))
    fig.add_trace(go.Scatter(x=pd.to_datetime(x, unit='s'), y=average_line_unrestrictive,
                            mode='lines',
                            name='Average Unrestrictive Trend',
                            line=dict(color='blue', width=6),))

    # Add buttons for toggling between different graphs
    buttons = [
        dict(label='All Countries',
            method='update',
            args=[{'visible': [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,False,False,False]}],
            name='All Countries'),
        dict(label='Restrictive Countries',
            method='update',
            args=[{'visible': [True,False,False,True,False,False,True,False,False,True,False,False,True,True,False,True,False,False]}]),
        dict(label='Semi-Restrictive Countries',
            method='update',
            args=[{'visible': [False,True,True,False,True,True,False,False,False,False,True,False,True,True,False,False,True,False]}]),
        dict(label='Unrestrictive Countries',
            method='update',
            args=[{'visible': [False,False,False,False,False,False,False,True,True,False,False,True,True,True,False,False,False,True]}]),
    ]

    fig.update_layout(
        title='Comparing Normalized Percentage of Wikipedia page views related to video games to English',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Normalized Percentage'),
        xaxis_range=[min(dates), max(dates)],
        xaxis_tickvals=pd.to_datetime(pd.date_range(start=dates[0], end=dates[-1], freq='90D')),
        xaxis_ticktext=pd.date_range(start=dates[0], end=dates[-1], freq='90D').strftime('%Y-%m-%d'),
        legend=dict(x=0, y=1),
        updatemenus=[{'type': 'buttons',
                    'showactive': True,
                    'buttons': buttons,
                    'active': 0,
                    'x': 1.3,
                    'y': 1}]
    )
    fig.update_traces(visible=False, selector=dict(name='Average Restrictive Trend'))
    fig.update_traces(visible=False, selector=dict(name='Average Semi-Restrictive Trend'))
    fig.update_traces(visible=False, selector=dict(name='Average Unrestrictive Trend'))

    fig.show()
    return

def plot_mobility(d, df_code, interventions):
    # Sample data
    # Assuming df_code, d, and interventions are defined

    # Create subplot
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Mobility'], vertical_spacing=0.1)

    # Define dropdown menu
    lockdown_types = ['Restrictive', 'Semi-Restrictive', 'Unrestrictive']
    lockdown_countries = {
        'Restrictive': ['fr', 'ca', 'it', 'sr'],
        'Semi-Restrictive': ['da', 'de', 'nl', 'no', 'fi', 'en'],
        'Unrestrictive': ['ko', 'ja', 'sv']
    }

    buttons = [dict(label=lockdown_type, method='update',
                    args=[{'visible': [c in lockdown_countries[lockdown_type] for c in df_code['lang']]}])
            for lockdown_type in lockdown_types]

    fig.update_layout(
        updatemenus=[dict(type='dropdown', active=0, buttons=buttons, x=0.1, y=1.15)],
    )

    for i, c in enumerate(df_code['lang']):
        dt = d[c]["topics"]["Culture.Media.Video games"]["percent"]
        dates = list(dt.keys())
        numbers = list(dt.values())
        dates = pd.to_datetime(dates)

        if c == 'sv':
            x = [datetime.timestamp(k) for k in dates]
            x = x[365:]
            y = [val for val in numbers if not np.isnan(val)]
        else:
            x = [datetime.timestamp(k) for k in dates]
            y = numbers

        # Creating the approximated curve
        degree = 5
        coefficients = np.polyfit(x, y, degree)
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(x)

        # Converting the mobility date (str) into a np.datetime64 to be able to use it
        mobility_g = interventions.loc[c]['Mobility']
        format_string = "%Y-%m-%d"
        date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

        # Offset each curve
        index = dates.get_loc(date_object)
        mean = y_fit[0:index].mean()
        offset = 0 - mean
        y_fit = y_fit + offset

        # Convert x back to datetime for plotting
        x_datetime = pd.to_datetime(x, unit='s')

        # Plot the trace
        fig.add_trace(go.Scatter(x=x_datetime, y=y_fit, mode='lines', name=c, visible=False))

    # Show the initial traces
    fig.data[0].visible = True
    fig.data[3].visible = True
    fig.data[6].visible = True
    fig.data[9].visible = True

    # Customize layout
    fig.update_layout(
        height=400,
        width=800,
        showlegend=True,
        legend=dict(x=1.02, y=1.0),  # Set x position greater than 1 to move legend to the right
        xaxis=dict(title='Date', tickangle=45, tickmode='array'),
        yaxis=dict(title='Percentage'),
    )

    fig.show()
    return

def return_game_figure(column_name, df, interventions, title='views'):

  fig, ax = plt.subplots(figsize=(15, 4))

  # Allows the user to enter 'Call of Duty' instead of 'Call_of_Duty'
  filtered_df = df[[column_name.replace(' ', '_')]]

  # Choose a color palette from seaborn
  color_palette = sns.color_palette("colorblind", len(filtered_df.columns))

  # Convert the color palette to a list
  list_of_colors = color_palette.as_hex()

  # Plotting each subcolumn corresponding to the language
  for column, color in zip(filtered_df.columns, list_of_colors):

      # Plots the number of views for each language with respect to time
      df_lang = interventions[interventions['lang']==column[1]]
      ax.plot(filtered_df.index, filtered_df[column], label=column[1], color=color)

      # Plots the period of lockdown in bold
      start_index = df_lang['Mobility'].item()
      end_index = df_lang['Normalcy'].item()
      limits_df = filtered_df.loc[start_index:end_index]
      ax.plot(limits_df.index, limits_df[column], color=color, linewidth=5)

  # Adding legend, labels, and title
  ax.set_yscale('log')
  ax.legend(loc='upper right',fontsize=7)
  ax.set_xlabel('Date')
  ax.set_ylabel('Views on the page (log scale)')
  if title == 'views':
    ax.set_title(f'Page Views Over Time for {column_name}')
  if title == 'percentage':
    ax.set_title(f'Percentage of Total Wikipedia Views for {column_name}')
  return fig

def return_specific_game():
    interventions = pd.read_csv('interventions.csv')
    country_code = ['en', 'fr', 'de', 'nl', 'fi', 'ja']

    start_dt = '2019100100' #Start day of the search
    end_dt = '2020123100' #End day of the search

    headers = {'User-Agent':'ADABot/0.0 (floydchow7@gmail.com)'}

    # Retrieve page views for the entire wikipedia for a particular country:

    df_wikiviews = pd.DataFrame()

    for country in country_code:
        # Declare f-string for all the different requests:
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/{country}.wikipedia/all-access/user/daily/{start_dt}/{end_dt}"
        try:
            r = requests.get(url, headers=headers)
            df_onequery = pd.DataFrame(r.json()['items'])
            df_wikiviews = pd.concat([df_wikiviews,df_onequery])
            time.sleep(0.5) # In case the IP address is blocked
        except:
            print('The {} page views are not found during these time'.format(country))

    # Drop useless columns and reset index
    df_wikiviews = df_wikiviews[['project', 'timestamp', 'views']].reset_index(drop=True)

    # Convert to timestamp to datetime variable
    df_wikiviews['timestamp'] = pd.to_datetime(df_wikiviews['timestamp'], format='%Y%m%d%H')

    # Rename the column from 'en.wikipedia' to 'en' and same for other languages
    df_wikiviews['project'] = df_wikiviews['project'].str.replace(r'\..*', '', regex=True)

    # Pivot the table to simplify further uses
    df_wikiviews = df_wikiviews.pivot_table(index = 'timestamp', columns = ['project'], values = 'views')

    main_url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/'

    # Name of the games you want to get the data of:
    games = ['Among Us', 'Fall Guys']

    df_gameviews = pd.DataFrame()

    for language in country_code:
        for game in games:
            try:
                url = main_url + language + '.wikipedia/all-access/user/' + game + '/daily/' + start_dt + '/' + end_dt
                r = requests.get(url,headers=headers)

                df_onequery = pd.DataFrame(r.json()['items'])
                df_gameviews = pd.concat([df_gameviews, df_onequery])

                time.sleep(0.5) # In case the IP address is blocked
            except:
                print('The {} page of {} is not found during these time'.format(language,game))
                
    # Keep only relevant columns and reset index:
    df_gameviews = df_gameviews[['project', 'article', 'timestamp','views']].reset_index(drop=True)

    # Convert timestamp to datetime format:
    df_gameviews['timestamp'] = pd.to_datetime(df_gameviews['timestamp'], format='%Y%m%d%H')

    # Rename the column from 'en.wikipedia' to 'en' and same for other languages
    df_gameviews['project'] = df_gameviews['project'].str.replace(r'\..*', '', regex=True)

    # Pivot table to have main column being the game, and subcolumns being the language`
    df_gameviews = df_gameviews.pivot_table(index = 'timestamp', columns = ['article', 'project'], values = 'views')

    # Rename columns
    df_gameviews.columns.set_names(['Game Name', 'Language'], level=[0, 1], inplace=True)

    # Filter the DataFrame to contain the data on a certain interval
    start_date = '2020-01-01'
    end_date = '2020-12-12'
    df_plot = df_gameviews.loc[start_date:end_date]

    fig, axes = plt.subplots(len(games), 1, figsize=(32, 20))

    # Loop through the subplots and create and plot a figure for each game
    for ax, game in zip(axes, games):

        fig_to_plot = return_game_figure(game, df_plot, interventions, 'views')

        sub_ax = fig_to_plot.get_axes()[0]
        sub_ax.get_figure().canvas.draw()

        buf = sub_ax.get_figure().canvas.renderer.buffer_rgba()
        plt.close(fig_to_plot)

        ax.imshow(buf)
        ax.axis('off')

    plt.xticks(rotation=45)  # Rotates the x-axis labels for better readability
    fig.tight_layout() # Adjusts the plot to ensure everything fits without overlapping
    plt.show()
    return


#Zhou
def save_topics_in_chunk(df_topiclinked):
    video_games_data = df_topiclinked[df_topiclinked['Culture.Media.Video games'] == True]
    game_topics = video_games_data['index'].str.replace('_',' ').values
    # Divide the whole topic datasets to implement multi-thread web-parsing in the website to increase efficiency.
    div = np.arange(0,game_topics.shape[0],4000)
    for i in range(len(div)):
        sub = game_topics[div[i]:div[i+1]-1] if i < len(div)-1 else game_topics[div[i]:]
        file_path = './game_topics/game_topic_'+str(i)+'.npy'
        if not os.path.exists(file_path):
            np.save(file_path,sub)  # Save the game topic datasets into seperated files
    return div

# save the crawled dats into the seperate files
def save_to_csv(data, num):
    file_name = f'./game_topics/game_topic_{str(num)}.csv'
    file_exists = os.path.isfile(file_name)
    with open(f'./game_topics/game_topic_{str(num)}.csv','a',newline='',encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        if not file_exists:
            csv_writer.writerow(['lang','topic','topic_in_English'])

        csv_writer.writerows(data)

# crawl the translated game topics in different languages
def crawl_title_lang(num):
    game_topics = np.load('./game_topics/game_topic_'+str(num)+'.npy',allow_pickle=True) #import the English-version game topics
    file_path = f'./game_topics/game_topic_{str(num)}.csv'
    if not os.path.exists(file_path):
        gamebar = tqdm(game_topics)

        # define the header setting for the parser
        count_dtpoint = 0
        chrome_options = Options()
        chrome_options.add_argument("User-Agent=ADABot/0.0 (floydchow7@gmail.com)")
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-extensions')
        
        # Parse the designated website
        for  game_topic in gamebar:
            title_languauge = []
            gamebar.set_description(f'Processing: {game_topic} with current title_languauge in length {str(count_dtpoint)}')
            url = 'https://pageviews.wmcloud.org/langviews/?project=en.wikipedia.org&platform=all-access&agent=user&range=latest-20&sort=views&direction=1&view=list&page='+game_topic
            English_name = game_topic
            try:
                driver = webdriver.Chrome(options=chrome_options)
                driver.get(url)
                wait = WebDriverWait(driver, 20)
                wait.until(
                    EC.visibility_of_element_located((By.ID,'output_list'))
                )
                html = driver.page_source
                driver.refresh()
                driver.delete_all_cookies()
                driver.close()
                driver.quit()

                # extract the needed elements from the website
                root = etree.HTML(html)
                names = root.xpath('//*[@id="output_list"]//tr//td//a[@dir="ltr"]//text()')
                langs = root.xpath('//*[@id="output_list"]//tr//td//a[@dir="ltr"]//@lang')

                # save the data into the files
                for lang, name in zip(langs, names):
                    title_languauge.append([lang, name, English_name])

                count_dtpoint = count_dtpoint + len(title_languauge)
                save_to_csv(title_languauge, num)
                del title_languauge
                del html
                gc.collect()


            except Exception as e:
                print(f"An error occured on {game_topic} : {(str(e)).split('Stacktrace')[0]}")
        try:
            driver.quit()
        except:
            pass
    #save_to_csv(title_languauge, num)
    #output = pd.DataFrame(title_languauge,columns=['lang','topic','topic_in_English'])
    #output.to_csv(f'./game_topics/topic_in_different_lang_{str(num)}_{str(index)}.csv')

# define the thread class for the data parsing
class titlecrawlThread(threading.Thread):
    def __init__(self, num):
        threading.Thread.__init__(self)
        self.num = num
    def run(self):
        crawl_title_lang(self.num)

def start_title_crawler_thread(div):
    thread_list = []
    for i in range(len(div)):
        thread = titlecrawlThread(i)
        thread.start()
        thread_list.append(thread)

    for thread in thread_list:
        thread.join()

# Crawl the pageviews of different datasets
def crawl_pageviews(thread_num,start_dt, end_dt):
    df_topics = pd.read_csv(f'./game_topics/game_topic_{str(thread_num)}.csv')
    eng_topics = list(set(df_topics['topic_in_English'].values))
    file_path = f'./pageviews/game_topic_{str(thread_num)}.csv'
    if not os.path.exists(file_path):
        loopbar = tqdm(eng_topics)
        headers = {'User-Agent':'ADABot/0.0 (floydchow7@gmail.com)'}
        df_wikiviews = pd.DataFrame()
        for eng_topic in loopbar:
            df_topic = df_topics[df_topics['topic_in_English']==eng_topic]
            loopbar.set_description(f"Processing {eng_topic} pageviews in {str(df_topic.shape[0])} language(s)")
            for index, row in df_topic.iterrows():
                lang = row['lang']
                topic = row['topic']
            # Declare f-string for all the different requests:
                url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{topic}/daily/{start_dt}/{end_dt}"
            try:
                r = requests.get(url, headers=headers)
                df_onequery = pd.DataFrame(r.json()['items'])
                df_onequery['topic'] = eng_topic
                df_wikiviews = pd.concat([df_wikiviews,df_onequery])
                time.sleep(0.5) # In case the IP address is blocked
                print(f'\r{" "*100}\rThe {eng_topic} pageviews in {lang} version found', end='', flush=True)
            except:
                print(f'\r{" "*100}\rThe {eng_topic} pageviews in {lang} version NOT found', end='', flush=True)
        

    # Convert to timestamp to datetime variable
        df_wikiviews['timestamp'] = pd.to_datetime(df_wikiviews['timestamp'], format='%Y%m%d%H')

    # Rename the column from 'en.wikipedia' to 'en' and same for other languages
        df_wikiviews['lang'] = df_wikiviews['project'].str.replace(r'\..*', '', regex=True)

        df_wikiviews = df_wikiviews[['topic','lang', 'timestamp', 'views',]].reset_index(drop=True)

        df_wikiviews.to_csv(f'./pageviews/game_topic_{str(thread_num)}.csv')
    
    return 0


class pageviewcrawlThread(threading.Thread):
    def __init__(self, thread_num, start_dt, end_dt):
        threading.Thread.__init__(self)
        self.thread_num = thread_num
        self.start_dt = start_dt
        self.end_dt = end_dt
    def run(self):
        crawl_pageviews(self.thread_num,self.start_dt, self.end_dt)

def start_pageview_crawler_thread(start_dt, end_dt):
    thread_list = []
    for i in range(9):
        thread = pageviewcrawlThread(i, start_dt, end_dt)
        thread.start()
        thread_list.append(thread)

    for thread in thread_list:
        thread.join()

def crawl_uncrawled_pageviews(df_topiclinked,thread_num,langs, start_dt, end_dt):
    file_path = f'./pageviews/game_topic_{str(thread_num+1)}.csv'
    if not os.path.exists(file_path):
    # Then we try to extract the untranslated game topics.
        video_games_data = df_topiclinked[df_topiclinked['Culture.Media.Video games'] == True]
        game_topics = video_games_data['index'].str.replace('_',' ').values
        uncrawled_topics = set(game_topics)
        for i in range(thread_num + 1):
            df_topics = pd.read_csv(f'./game_topics/game_topic_{str(i)}.csv')
            crawled_topics = set(df_topics['topic_in_English'].values)
            uncrawled_topics = uncrawled_topics - crawled_topics
        # Then we try to extract uncrawled_topic and form posudo-api links since we didn't know the actual translation in different languages
        print(f"There is {str(len(uncrawled_topics))} topics need to be crawled")
        headers = {'User-Agent':'ADABot/0.0 (floydchow7@gmail.com)'}
        df_wikiviews = pd.DataFrame()
        
        loopbar = tqdm(list(uncrawled_topics))
        for uncrawled_topic in loopbar:
            loopbar.set_description(f"Processing {uncrawled_topic} pageviews")
            for lang in langs:        
                url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/{uncrawled_topic}/daily/{start_dt}/{end_dt}"
                try:
                    r = requests.get(url, headers=headers)
                    df_onequery = pd.DataFrame(r.json()['items'])
                    df_onequery['topic'] = uncrawled_topic
                    df_wikiviews = pd.concat([df_wikiviews,df_onequery])
                    time.sleep(0.5) # In case the IP address is blocked
                    print(f'\r{" "*100}\rThe {uncrawled_topic} pageviews in {lang} version found', end='', flush=True)
                except:
                    pass

            # Convert to timestamp to datetime variable
        df_wikiviews['timestamp'] = pd.to_datetime(df_wikiviews['timestamp'], format='%Y%m%d%H')

        # Rename the column from 'en.wikipedia' to 'en' and same for other languages
        df_wikiviews['lang'] = df_wikiviews['project'].str.replace(r'\..*', '', regex=True)

        df_wikiviews = df_wikiviews[['topic','lang', 'timestamp', 'views',]].reset_index(drop=True)

        df_wikiviews.to_csv(f'./pageviews/game_topic_{str(thread_num+1)}.csv')
    

# Now we try to extract the categories for each wikidata
def extract_game_genre(thread_num):
    file_path = f'./game_genres/game_genres_{str(thread_num)}.csv'
    if not os.path.exists(file_path):
        raw_gametopic_df = pd.read_csv('./game_topics/raw_gametopic_data.csv')
        game_topic_df = pd.read_csv(f'./pageviews/game_topic_{str(thread_num)}.csv')
        game_topic_df = set(game_topic_df['topic'])
        selected_gametopic_df = raw_gametopic_df[raw_gametopic_df['index'].isin(game_topic_df)].copy()
        selected_gametopic_df['genres'] = pd.NA
        # Define url for query
        endpoint_url = "https://query.wikidata.org/sparql"
        headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json'
            }
        for index, row in tqdm(selected_gametopic_df.iterrows(), total=len(selected_gametopic_df), desc="Processing rows"):
            qid = row['qid']
            query = """
            SELECT ?genreLabel
            WHERE {
                wd:""" + qid + """ wdt:P136 ?genre.
                SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
            }
            """
            response = requests.get(endpoint_url, params={'query': query, 'format': 'json'}, headers=headers)
            data = response.json()
            # extract the genres in the wikipidea pages for every game topic
            genres = [item['genreLabel']['value'] for item in data['results']['bindings']] if 'results' in data else []
            selected_gametopic_df.at[index, 'genres'] = genres
            time.sleep(0.5)
        selected_gametopic_df.to_csv(f'./game_genres/game_genres_{str(thread_num)}.csv',index=False,encoding='utf-8')
    return None

class genrecrawlThread(threading.Thread):
    def __init__(self, thread_num):
        threading.Thread.__init__(self)
        self.thread_num = thread_num
    def run(self):
        extract_game_genre(self.thread_num)

def start_genre_crawler_thread():
    thread_list = []
    for i in range(10):
        thread = genrecrawlThread(i)
        thread.start()
        thread_list.append(thread)

    for thread in thread_list:
        thread.join()

def filter_game_genres(raw_game_filepath):
    game_df = pd.read_csv(raw_game_filepath)
    # We only extract the topics that are actually games, which will have the unempty genres columns
    df = game_df.copy(deep=True)
    df['genres'].apply(lambda x: len(x)>2)
    df = df.loc[df['genres'].apply(lambda x: len(x)>2),['index','genres']]
    df['genres'] = df['genres'].apply(lambda x: x.replace("'","").replace("\"","").replace("[","").replace("]","").split(","))
    new_df = pd.DataFrame(columns=['index','genres'])

    # Split the multiple genres into different rows
    for index, row in df.iterrows():
        game = row['index']
        genre_list = row['genres']
        for genre in genre_list:
            new_df.loc[len(new_df.index)] = [game, genre]

    # We check whether there is a tab in the genres columns and convert it to normal one
    new_df['genres'] = new_df['genres'].apply(lambda x: x[1:] if x[0]==' 'else x)

    # count the games in different genres
    count_genres_df = new_df.groupby(['genres'],as_index=False).agg({'index':'count'}).sort_values('index',ascending=False).reset_index(drop=True)
    count_genres_df.columns = ['genres','count']

    # Then we aggergate the count_df with the genre_df to obtain the main genres(which means the highest genres) in the datasets
    new_df2 = pd.merge(new_df,count_genres_df,on='genres',how='left')

    # rank the raw game genres
    tmp_df = new_df2.groupby('index',as_index=False).apply(lambda x: x.sort_values(by='count',ascending=False))[['index','genres','count']].reset_index(drop=True)
    tmp_df['rank'] = tmp_df.groupby('index').cumcount() + 1
    tmp_df['rank'] = tmp_df['rank'].apply(lambda x: 'genre '+ str(x))
    
    # alternate the dataset into the pivot table
    tmp_pivot_df = tmp_df.pivot(index='index',columns='rank',values='genres').reset_index()
    reorder_columns = ['genre '+str(i) for i in np.arange(1,14,1)]
    reorder_columns.insert(0,'index')
    tmp_pivot_df = tmp_pivot_df[reorder_columns]

    # We only obtain the genres with the highest count as the main genres
    result_df = new_df2.copy(deep=True)
    return result_df

def main_genre_classification(raw_classification_filepath, result_df):
    gpt_classification_df = pd.read_csv(raw_classification_filepath)
    gpt_classification_df = gpt_classification_df[['Small Game Genres','Larger Game Genres']]
    gpt_classification_df['Small Game Genres'] = gpt_classification_df['Small Game Genres'].apply(lambda x: x.lower().replace("'",""))
    gpt_classification_df['Larger Game Genres'] = gpt_classification_df['Larger Game Genres'].apply(lambda x: x.split(",")[0])

    #Explore the chatGPT-classification in the main Genres
    gpt_classification_df.columns = ['genres','Main Genre']
    main_genre_df = pd.merge(result_df, gpt_classification_df, on='genres', how='left')

    main_genre_df = main_genre_df[['index','Main Genre','genres']]
    main_genre_df.columns = ['Game','Main Genre','Secondary Genre']

    main_genre_df.to_csv('./Milestone3/gpt-classification.csv',encoding='utf-8',index=False)
    return main_genre_df

def display_main_genre_stats(main_genre_df):
    stats_df = main_genre_df.drop_duplicates(subset=['Game','Main Genre']).groupby("Main Genre",as_index=False).agg({"Game":"count"}).sort_values("Game",ascending=False).reset_index(drop=True)
    return stats_df

def visualize_genres_distribution(stats_df, others_threadshold):
    # Create a new DataFrame for 'Others' category
    df_others = stats_df[stats_df['Game'] <= others_threadshold]
    others_row = pd.DataFrame({'Main Genre': ['Others'], 'Game': [df_others['Game'].sum()]})
    stats_df = pd.concat([stats_df[stats_df['Game'] > others_threadshold], others_row], ignore_index=True)

    # Sort the DataFrame by 'Game' column in descending order
    stats_df = stats_df.sort_values(by='Game', ascending=True)

    # Plot using Plotly Express
    fig = px.bar(stats_df, x='Game', y='Main Genre', orientation='h', color='Game',
                 labels={'Game': 'Number of Games', 'Main Genre': 'Main Genre'},
                 title='Game Genres Distribution',
                 text='Game', height=500)

    # Customize layout
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False)

    html_file_path = "game genre count.html"
    pio.write_html(fig, file=html_file_path)

    # Show the plot
    fig.show()


def visualize_pageviews_in_genre(pageviews_filepath, genres_filepath):
    pageviews = pd.read_csv(pageviews_filepath)
    game_genres = pd.read_csv(genres_filepath)
    pageviews.columns = ['Game', 'lang', 'timestamp', 'views']
    merged_df = pd.merge(pageviews, game_genres, on='Game', how='left')
    merged_df.dropna(inplace=True)
    grouped_df = merged_df.groupby(by=['Main Genre', 'timestamp', 'lang'], as_index=False).agg(pageviews=pd.NamedAgg(column='views', aggfunc='sum'))

    # We visualize the total pageviews according to the game genres on some main languages except English
    main_genres = list(set(grouped_df['Main Genre']))
    main_genres.remove('Comics')

    # Create a single subplot
    fig = make_subplots(rows=1, cols=1, subplot_titles=['Game Genres Pageviews'], shared_xaxes=True, shared_yaxes=True)

    # Create traces for each genre
    traces = []

    for genre in main_genres:
        sub_grouped_df = grouped_df[(grouped_df['Main Genre'] == genre) & (grouped_df['lang'].isin(['de', 'fr', 'it', 'pt', 'es', 'ja']))]
        sub_grouped_df = sub_grouped_df.copy()
        sub_grouped_df['timestamp'] = pd.to_datetime(sub_grouped_df['timestamp'])

        for lang in sub_grouped_df['lang'].unique():
            lang_data = sub_grouped_df[sub_grouped_df['lang'] == lang]
            trace = go.Scatter(x=lang_data['timestamp'], y=lang_data['pageviews'], mode='lines', name=f'{genre} | {lang}', showlegend=True)
            traces.append(trace)
            fig.add_trace(trace)

        # Add a hidden trace for each genre
        #hidden_trace = go.Scatter(x=lang_data['timestamp'], y=np.zeros(len(lang_data['pageviews'])), name='', showlegend=True)
        #traces.append(hidden_trace)
        #fig.add_trace(hidden_trace)

    # Customize layout
    fig.update_layout(title_text='Game Genres Pageviews', height=600, legend=dict(x=-0.3, y=0.5, font=dict(size=8)))

    # Add dropdown menu to select different main genres
    dropdown_buttons = [
        {'label': genre, 'method': 'update', 'args': [{'visible': [genre == trace.name.split(' | ')[0] for trace in traces]}]}
        for genre in main_genres
    ]

    fig.update_layout(updatemenus=[{'active': 0, 'buttons': dropdown_buttons, 'showactive': True}])


    html_file_path = f"pageviews.html"
    pio.write_html(fig, file=html_file_path)
    # Show the plot
    fig.show()

def convert_to_code_dict(df_code):
    #convert it the dictionary
    code_dict = dict(zip(df_code['lang'],df_code['state']))
    return code_dict

def merge_mobility_pageview(globalmob, pageviews, game_genres, code_dict):

    #Align the pageviews and categories
    pageviews.columns = ['Game','lang','timestamp','views']
    merged_df = pd.merge(pageviews, game_genres,on='Game',how='left')
    merged_df.dropna(inplace=True)
    grouped_df = merged_df.groupby(by=['Main Genre','timestamp','lang'],as_index=False).agg(pageviews = pd.NamedAgg(column='views',aggfunc='sum'))
    grouped_df = grouped_df.replace({'lang': code_dict})
    lang_pageviews_df = grouped_df.groupby(by=['lang','timestamp'],as_index=False).agg(pageviews = pd.NamedAgg(column='pageviews',aggfunc='sum'))
    lang_pageviews_df['lang'] = lang_pageviews_df['lang'].apply(lambda x: x.upper())
    lang_pageviews_df.columns = ['country_region_code','date','pageviews']
    # We change pageviews to baseline change
    baseline = '2020-02-14' #Define it as the baseline for the pageviews
    lang_pageviews_df = lang_pageviews_df[lang_pageviews_df['date']>=baseline]

    baseline_pageviews = lang_pageviews_df[lang_pageviews_df['date']==baseline]
    baseline_pageviews.columns = ['country_region_code','date','baseline pageviews']
    baseline_pageviews.drop(['date'], axis=1,inplace=True)
    lang_pageviews_df = pd.merge(lang_pageviews_df,baseline_pageviews,on='country_region_code',how='left')
    lang_pageviews_df['change from baseline'] = 100*(lang_pageviews_df['pageviews']/lang_pageviews_df['baseline pageviews']-1) # Calculate the change compared to baseline pageview in different languages

    globalmob = pd.merge(globalmob, lang_pageviews_df,on=['country_region_code','date'],how='left') #Merged with global mobility datasets
    globalmob.drop(['pageviews','baseline pageviews'],axis=1, inplace=True)
    return grouped_df, globalmob

def visualize_mobility_pageviews(globalmob, interventions, df_code):
    # Create a subplot with Plotly
    fig = make_subplots(rows=(len(df_code['lang'])//2)+1, cols=2, shared_yaxes=False, subplot_titles=df_code.index.tolist(), vertical_spacing=0.1)

    for i, c in enumerate(df_code['lang']):
        cs = df_code.iloc[i]['state']

        if cs == 'KR':
            globalmob_ko = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull()) & (globalmob['metro_area'].isnull())]
        else:
            if cs == 'RS':
                globalmob_sr = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['metro_area'].isnull())]
            else:
                if cs == 'ES':
                    globalmob_ca = globalmob[(globalmob['country_region_code'] == 'ES') & (globalmob['sub_region_1'] == 'Catalonia') & (globalmob['sub_region_2'].isnull())].copy()
                else:
                    globalmob_g = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull())].copy()
                    globalmob_g.reset_index(inplace=True, drop=True)

        df = globalmob_g.copy(deep=True)

        mobility_g = interventions.loc[c]['Mobility']
        lockdown_g = interventions.loc[c]['Lockdown']
        normalcy_g = interventions.loc[c]['Normalcy']

        columns = globalmob.columns[8:]
        df = df.drop(['residential_percent_change_from_baseline', 'parks_percent_change_from_baseline'], axis=1)
        columns = columns.drop(['residential_percent_change_from_baseline', 'parks_percent_change_from_baseline'])

        mean_g = df[columns.drop(['change from baseline'])].mean(axis=1)

        row = i // 2 + 1
        col = i % 2 + 1

        # Create traces for the plot
        mean_trace = go.Scatter(x=df['date'], y=mean_g, mode='lines', name='Average change in mobility', line=dict(color='blue'))
        pageview_trace = go.Scatter(x=df['date'], y=df['change from baseline'], mode='lines', name='Change in pageviews in Games', line=dict(color='red'))

        # Add traces to the subplot
        fig.add_trace(mean_trace, row=row, col=col)
        fig.add_trace(pageview_trace, row=row, col=col)


        # Add vertical lines for events if not NA
        if not pd.isna(lockdown_g):
            fig.add_shape(type="line", x0=lockdown_g, x1=lockdown_g, y0=-100, y1=200, line=dict(color="black", width=2.2, dash="dash"), row=row, col=col)

        if not pd.isna(mobility_g):
            fig.add_shape(type="line", x0=mobility_g, x1=mobility_g, y0=-100, y1=200, line=dict(color="blue", width=1.5, dash="solid"), row=row, col=col)

        if not pd.isna(normalcy_g):
            fig.add_shape(type="line", x0=normalcy_g, x1=normalcy_g, y0=-100, y1=200, line=dict(color="black", width=1.5, dash="solid"), row=row, col=col)


        # Update x-axis ticks
        tickvals = [13, 42, 73, 103, 134, 164]
        ticktext = ['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul']

        #fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, row=row, col=col)


    # Show the plot
    fig.update_layout(title_text="Change of mobility and game pageviews in different countries in Year 2020(%)", showlegend=False)
    fig.update_layout(height=1200, width=1000)  # Adjust the figure size as needed
    html_file_path = "pageviews and countries.html"
    pio.write_html(fig, file=html_file_path)

    fig.show()

def visualize_different_language(lang, baseline, interventions, grouped_df, globalmob, code_dict, country_name, omit_genre=[]):
    # Extract data
    pageviews_sub = grouped_df[(grouped_df['lang'] == lang.upper()) & (grouped_df['timestamp'] >= baseline)]
    baseline_pageviews_sub = pageviews_sub[pageviews_sub['timestamp'] == baseline].drop(['lang', 'timestamp'], axis=1)
    pageviews_sub.columns = ['Main Genre', 'date', 'lang', 'pageviews']
    baseline_pageviews_sub.columns = ['Main Genre', 'baseline pageviews']
    pageviews_sub = pd.merge(pageviews_sub, baseline_pageviews_sub, on=['Main Genre'], how='left')
    pageviews_sub['change from basetime'] = 100 * (pageviews_sub['pageviews'] / pageviews_sub['baseline pageviews'] - 1)
    pageviews_sub['lang'] = pageviews_sub['lang'].apply(lambda x: x.upper())
    pageviews_sub_result = pd.pivot_table(pageviews_sub, values='change from basetime', index='date',
                                          columns='Main Genre').reset_index().dropna(axis=1)

    globalmob_g = globalmob[(globalmob['country_region_code'] == lang.upper()) & (globalmob['sub_region_1'].isnull())].drop(
        ['change from baseline'], axis=1).dropna(axis=1)

    df = pd.merge(globalmob_g, pageviews_sub_result, on='date', how='left')

    matching_lang = [key for key, value in code_dict.items() if value == lang.upper()]

    mobility_fr = interventions.loc[matching_lang[0]]['Mobility']
    lockdown_fr = interventions.loc[matching_lang[0]]['Lockdown']
    normalcy_fr = interventions.loc[matching_lang[0]]['Normalcy']

    columns = df.columns[4:].copy()

    mean_fr = df.loc[:, columns.drop(pageviews_sub_result.columns.drop(['date']))].mean(axis=1)

    selected_genres = pageviews_sub_result.columns.drop(['date'] + omit_genre) if len(omit_genre) > 0 else pageviews_sub_result.columns.drop(['date'])

    # Create Plotly subplot
    fig = make_subplots(rows=(len(selected_genres) + 1) // 2, cols=2,
                        subplot_titles=list(selected_genres))
    fig.update_annotations(font_size=10)
    fig.update_layout(font=dict(size=8))
    fig.update_layout(margin=dict(l=0,r=6,b=4), mapbox_style = "open-street-map")
    for i, genre in enumerate(selected_genres):
        row = i // 2 + 1
        col = i % 2 + 1
        genre_trace = go.Scatter(x=df['date'], y=df[genre], mode='lines', name=genre, line=dict(color='red',width=1))
        fig.add_trace(genre_trace, row=row, col=col)

        mean_trace = go.Scatter(x=df['date'], y=mean_fr, mode='lines', name='Average percentage change in mobility' , line=dict(color='blue',width=1))
        fig.add_trace(mean_trace, row=row, col=col)
    # Add vertical lines for events if not NA
        if not pd.isna(lockdown_fr):
            lockdown_shape = dict(type="line", x0=lockdown_fr, x1=lockdown_fr, y0=-100,y1=max(df[genre]),
                                line=dict(color="black", width=1, dash="dash"))
            fig.add_shape(lockdown_shape, row=row, col=col)

        if not pd.isna(mobility_fr):
            mobility_shape = dict(type="line", x0=mobility_fr, x1=mobility_fr, y0=-100,y1=max(df[genre]),
                                line=dict(color="blue", width=1, dash="solid"))
            fig.add_shape(mobility_shape, row=row, col=col)

        if not pd.isna(normalcy_fr):
            normalcy_shape = dict(type="line", x0=normalcy_fr, x1=normalcy_fr, y0=-100,y1=max(df[genre]),
                                line=dict(color="black", width=1, dash="solid"))
            fig.add_shape(normalcy_shape, row=row, col=col)


    # Update layout
    fig.update_layout(
        title_text=f'Mobility and attention shift in different game genres in {country_name}(%)',
        showlegend=False,
        height=800, 
        width=500, 
    )

    # Save the figure as an HTML file
    html_file_path = f"{country_name.lower()}.html"
    pio.write_html(fig, file=html_file_path)

    # Display the HTML link
    fig.show()
    # Define the function to visualize the pageviews and mobilities change in different game genres
    """_summary_

    Args:
        lang : the language we analyze
        grouped_df : the pageviews datasets grouped from game genres
        globalmob : the global mobilites function
        code_dict (_type_): country code dictionary
        omit_genre (optional): The game genre to omit during analysis
    """
    pageviews_sub = grouped_df[(grouped_df['lang']==lang.upper())&(grouped_df['timestamp']>=baseline)]
    baseline_pageviews_sub = pageviews_sub[pageviews_sub['timestamp']==baseline].drop(['lang','timestamp'],axis=1)
    pageviews_sub.columns=['Main Genre','date','lang','pageviews']
    baseline_pageviews_sub.columns = ['Main Genre','baseline pageviews']
    pageviews_sub = pd.merge(pageviews_sub, baseline_pageviews_sub,on=['Main Genre'],how='left')
    pageviews_sub['change from basetime'] = 100*(pageviews_sub['pageviews']/pageviews_sub['baseline pageviews']-1)
    pageviews_sub['lang'] = pageviews_sub['lang'].apply(lambda x: x.upper())
    pageviews_sub_result = pd.pivot_table(pageviews_sub,values='change from basetime', index='date',columns='Main Genre').reset_index().dropna(axis=1)

    globalmob_g = globalmob[(globalmob['country_region_code'] == lang.upper()) & (globalmob['sub_region_1'].isnull())].drop(['change from baseline'],axis=1).dropna(axis=1)

    df = pd.merge(globalmob_g, pageviews_sub_result, on='date',how='left')

    #selected_genres = ['Action', 'Adult',
    #    'Adventure', 'Anime/Manga', 'Fantasy', 'Horror',
    #    'Multiplayer/Online', 'Puzzle', 'Racing',
    #        'Sports', 'Strategy']
    matching_lang = [key for key, value in code_dict.items() if value == lang.upper()]

    mobility_fr = interventions.loc[matching_lang[0]]['Mobility']
    lockdown_fr = interventions.loc[matching_lang[0]]['Lockdown']
    normalcy_fr = interventions.loc[matching_lang[0]]['Normalcy']

    columns = df.columns[4:].copy()

    mean_fr = df.loc[:, columns.drop(pageviews_sub_result.columns.drop(['date']))].mean(axis=1)

    selected_genres = pageviews_sub_result.columns.drop(['date']+ omit_genre) if len(omit_genre)>0 else pageviews_sub_result.columns.drop(['date'])

    #fig, axs = plt.subplots(len(pageviews_sub_result.columns.drop(['date']))//2, 2, sharey=True, figsize=(20, 20))
    fig, axs = plt.subplots((len(selected_genres)+1)//2, 2, sharey=True, figsize=(20, 20))

    for i, genre in enumerate(selected_genres):

        row = i // 2
        col = i % 2
        ax = axs[row, col]
        mean_line, = ax.plot(df['date'], mean_fr, label='Average percentage change in mobility')
        for column in columns:
            if column in pageviews_sub_result.columns:
                if column in [genre]:
                    genre_line, = ax.plot(df['date'], df[column], label=column,color='red',linestyle='--')
                elif column not in omit_genre:
                    ax.plot(df['date'], df[column], label=column, color='red',linestyle='--', alpha=0.1)
                else:
                    pass
            else:
                ax.plot(df['date'], df[column], label=column, color='black', alpha=0.1)

        ax.axvline(lockdown_fr, color='black', lw=2.2, linestyle="--")
        ax.axvline(mobility_fr, color='blue', lw=1.5, linestyle="-", alpha=0.7)
        ax.axvline(normalcy_fr, color='black', lw=1.5, linestyle="-", alpha=0.5)
        ax.set_xticks([13, 42, 73, 103, 134, 164], ['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul'])
        ax.set_xlim(min(df['date']), max(df['date']))
        ax.grid(True)
        ax.legend(handles=[mean_line,genre_line],loc='upper right')
    plt.suptitle(f'Mobility and attention shift in different game genres in {country_name}',fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    plt.show()

# Coco

def defglobalmob(cs, globalmob):
    match cs:

        case 'KR':
          globalmob_ko = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull()) & (globalmob['metro_area'].isnull())].copy()
          df = globalmob_ko

        case 'RS':
          globalmob_sr = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['metro_area'].isnull())].copy()
          globalmob_sr.reset_index(inplace=True)
          globalmob_sr.drop('index', axis=1, inplace=True)

          date_range = pd.date_range('2020-05-19', '2020-07-02', normalize=True)
          date_range = date_range.date
          date_range_str = [d.strftime("%Y-%m-%d") for d in date_range]

          # Rows to be inserted
          num_rows_to_insert = 45
          new_rows_data = {'country_region_code': 'RS',
                          'country_region': 'Serbia',
                          'sub_region_1': [np.nan] * num_rows_to_insert,
                          'sub_region_2': [np.nan] * num_rows_to_insert,
                          'metro_area': [np.nan] * num_rows_to_insert,
                          'iso_3166_2_code': [np.nan] * num_rows_to_insert,
                          'census_fips_code': [np.nan] * num_rows_to_insert,
                          'date': date_range_str,
                          'retail_and_recreation_percent_change_from_baseline': -15.25,
                          'grocery_and_pharmacy_percent_change_from_baseline': -15.25,
                          'transit_stations_percent_change_from_baseline': -15.25,
                          'workplaces_percent_change_from_baseline': -15.25
          }
          new_rows_df = pd.DataFrame(new_rows_data)

          # Index where you want to insert the new rows (in this case, after the second row)
          index_to_insert = 93

          # Insert the new rows
          merged = pd.concat([globalmob_sr.loc[:index_to_insert], new_rows_df, globalmob_sr.loc[index_to_insert+1:]]).reset_index(drop=True)
          globalmob_sr = merged
          df = merged

        case 'ES':
          globalmob_ca = globalmob[(globalmob['country_region_code'] == 'ES') & (globalmob['sub_region_1'] == 'Catalonia') & (globalmob['sub_region_2'].isnull())].copy()
          df = globalmob_ca

        # If an exact match is not confirmed, this last case will be used if provided
        case _:
          globalmob_g = globalmob[(globalmob['country_region_code'] == cs) & (globalmob['sub_region_1'].isnull())].copy()
          globalmob_g.reset_index(inplace=True, drop=True)
          df = globalmob_g.copy(deep=True)

    df.drop(['residential_percent_change_from_baseline', 'parks_percent_change_from_baseline'], axis=1, inplace=True)
    df = df.reset_index(drop=True)
    columns = df.columns[8:]
    mean_g = df[columns].mean(axis=1)
    mean_g = mean_g.reset_index(drop=True)
    return df, mean_g

def plotmob_inter(globalmob, df_code, interventions):
  fig = make_subplots(rows=6, cols=2, shared_xaxes=True, shared_yaxes=True,
                          horizontal_spacing = 0.025, vertical_spacing = 0.025,
                          subplot_titles=("Mobility in " + df_code.index))

  for i, c in enumerate(df_code['lang']):
      cs = df_code.iloc[i]['state']
      df, mean_g = defglobalmob(cs, globalmob)

      mobility_g = interventions.loc[c]['Mobility']
      lockdown_g = interventions.loc[c]['Lockdown']
      normalcy_g = interventions.loc[c]['Normalcy']

      x = df['date'].index
      y = mean_g
      poly = np.polyfit(x, y, 10)
      poly_y = np.poly1d(poly)(x)


      row = i // 2 +1
      col = i % 2 +1

      n = "Approximation for " + df_code.index[i]
      s = False
      if cs == 'JP':
        s = True

        # Add vertical lines at x = 2 and x = 4
      fig.add_trace(go.Scatter(x=df['date'], y=mean_g, mode='lines', name='Line 1', line=dict(color='rgba(0,0,205,1)', width=1), showlegend = False), row=row, col=col)
      fig.add_trace(go.Scatter(x=df['date'], y=poly_y, mode='lines', line=dict(color='rgba(255,140,0,1)', width=1),
                               hoverinfo='skip', legendgroup="group", name= 'Approximation curves',
                               showlegend = s, visible='legendonly'), row=row, col=col)
      for column in df.columns[8:]:
          fig.add_trace(go.Scatter(x=df['date'], y=df[column], mode='lines', line=dict(color='rgba(0,0,0,0.1)', width=1),
                                   hoverinfo='skip', showlegend = False), row=row, col=col)

      fig.update_annotations(font_size=12)
      fig.update_layout(font=dict(size=8))
      fig.update_yaxes(showgrid=True, range=[-100, 50], dtick=20, title_text='change (%)', row=row, col=col)
      fig.update_xaxes(title_text='Date', row=row, col=col)
      fig.add_shape(type='line', x0=mobility_g, x1=mobility_g, y0=-100, y1=50, line=dict(color='red', width=1.5), row=row, col=col)
      fig.add_shape(type='line', x0=normalcy_g, x1=normalcy_g, y0=-100, y1=50 , line=dict(color='red', width=1.5, dash='dot'), row=row, col=col)

  fig.update_layout(
    margin=dict(l=10, r=10, t=50, b=0),
    legend=dict(x=0, y=-0.1, traceorder='normal', orientation='h'),
    )

  fig.show()

def plotmob(globalmob, df_code, interventions):
  fig, axs = plt.subplots((len(interventions)-1)//2, 2, sharey=True, figsize=(20, 20))

  for i, c in enumerate(df_code['lang']):
      cs = df_code.iloc[i]['state']
      df, mean_g = defglobalmob(cs, globalmob)

      mobility_g = interventions.loc[c]['Mobility']
      lockdown_g = interventions.loc[c]['Lockdown']
      normalcy_g = interventions.loc[c]['Normalcy']

      x = df['date'].index
      y = mean_g
      poly = np.polyfit(x, y, 10)
      poly_y = np.poly1d(poly)(x)


      row = i // 2
      col = i % 2

      axs[row, col].plot(df['date'], mean_g)
      axs[row, col].plot(x, poly_y)
      for column in df.columns[8:]:
          axs[row, col].plot(df['date'], df[column], label=column, color='black', alpha=0.1)

      axs[row, col].axvline(lockdown_g, color='black', lw=2.2, linestyle="--")
      axs[row, col].axvline(mobility_g, color='blue', lw=1.5, linestyle="-", alpha=0.7)
      axs[row, col].axvline(normalcy_g, color='black', lw=1.5, linestyle="-", alpha=0.5)

      axs[row, col].set_xticks([13, 42, 73, 103, 134, 164], ['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul'])
      axs[row, col].set_xlim(min(df['date']), max(df['date']))
      axs[row, col].grid(True)
      axs[row, col].set_title('Mobility in ' + df_code.index[i])
      axs[row, col].set_xlabel('date')
      axs[row, col].set_ylabel('percentage of mobility compared to day 0')

      lines = [
        Line2D([0], [0], color='gray', alpha=0.5, linestyle='-'),
        Line2D([0], [0], color='tab:blue', linewidth=2, linestyle='-'),
        Line2D([0], [0], color='blue', linewidth=1.5, linestyle='-'),
        Line2D([0], [0], color='black', lw=2.2, linestyle="--"),
        Line2D([0], [0], color='gray', linewidth=1.5, linestyle='-')
        ]
      lines_labels = ['Mobility Signals',
                'Mean for each country',
                'Mobility change point',
                'Start of the lockdown',
                'Normalcy date'
                ]
      fig.legend(lines, lines_labels, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.025), frameon=False, fontsize=12)


  plt.tight_layout()
  plt.show()

def meanmob(df_code, globalmob):
  df_mean = pd.DataFrame()

  for i, c in enumerate(df_code['lang']):
    cs = df_code.iloc[i]['state']
    _,mean_g = defglobalmob(cs, globalmob)
    df_mean[df_code.index[i]] = mean_g

  return df_mean

def meanmobplot(df_code, interventions, globalmob):
  df_mean = meanmob(df_code, globalmob)
  df_code['mean_mob'] = None

  for i, c in enumerate(df_code['lang']):

    cs = df_code.iloc[i]['state']
    country = df_code.index[i]
    m = interventions.loc[c]['Mobility']
    n = interventions.loc[c]['Normalcy']
    mob, _ = defglobalmob(cs, globalmob)
    index_m = (mob['date'] == m).index[mob['date'] == m].tolist()[0]
    index_n = (mob['date'] == n).index[mob['date'] == n].tolist()[0]

    mean_value = df_mean[country].iloc[index_m:index_n+1].mean()
    df_code.loc[country]['mean_mob'] = mean_value

  ax = df_code['mean_mob'].plot.bar(figsize=(7, 6))
  ax.grid()  # grid lines
  ax.set_axisbelow(True)

  # Add overall title
  plt.title('Average decrease in mobility depending on the country')

  # Add axis titles
  plt.xlabel('country/region')
  plt.xticks(rotation=45);
  plt.ylabel('Average decrease in mobility [%]')
  plt.grid(True)
  plt.gca().invert_yaxis()

  # Show the plot
  plt.show()

def smoothedmobility(df_code, globalmob):
  fig, axs = plt.subplots(len(df_code)//2, 2, sharex = True, sharey = True, figsize=(20, 20))

  for i, c in enumerate(df_code['lang']):

    country_code = df_code.iloc[i]['state']
    country,av = defglobalmob(country_code, globalmob)

    x = country['date'].index
    y = av
    poly = np.polyfit(x, y, 10)
    poly_y = np.poly1d(poly)(x)

    row = i // 2
    col = i % 2
    axs[row, col].plot(x, y)
    axs[row, col].plot(x, poly_y)

    axs[row, col].set_xticks([13, 42, 73, 103, 134, 164])
    axs[row, col].set_xticklabels(['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul'])

    axs[row, col].set_xlim(min(x), max(x))

    axs[row, col].grid(True)
    axs[row, col].set_title('Mobility in ' + df_code.index[i])
    axs[row, col].set_xlabel('date')
    axs[row, col].set_ylabel('percentage of mobility compared to day 0')

  plt.tight_layout()
  plt.show()

def aggregatedmobilitysmoothedplot(df_code, globalmob):
  for i, c in enumerate(df_code['lang']):

    country_code = df_code.iloc[i]['state']
    _,av = defglobalmob(country_code, globalmob)
    country,_ = defglobalmob(country_code, globalmob)

    x = country['date'].index
    y = av
    poly = np.polyfit(x, y, 10)
    poly_y = np.poly1d(poly)(x)
    plt.plot(x, poly_y, label=c)

  plt.xticks([13, 42, 73, 103, 134, 164], ['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul'])
  plt.xlim(min(x), max(x))

  plt.grid(True)
  plt.title('Mobility depending on the country')
  plt.xlabel('date')
  plt.ylabel('percentage of mobility compared to day 0')

  plt.legend()
  plt.show()

def desaggregatedmobilitysmoothed(df_code, globalmob):
  fig, axs = plt.subplots(1, 3, sharex = True, sharey = True, figsize=(12,5))

  for i, c in enumerate(df_code['lang']):

    country_code = df_code.iloc[i]['state']
    _,av = defglobalmob(country_code, globalmob)
    country,_ = defglobalmob(country_code, globalmob)

    x = country['date'].index
    y = av
    poly = np.polyfit(x, y, 10)
    poly_y = np.poly1d(poly)(x)

    row=0
    if c in ['fr', 'ca', 'it', 'sr']:
      col = 0
    else:
      if c in ['ko', 'ja', 'sv']:
        col = 2
      else:
        col = 1

    axs[col].plot(x, poly_y, label=c)

    axs[col].set_xticks([13, 42, 73, 103, 134, 164])
    axs[col].set_xticklabels(['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul'])

    axs[col].set_xlim(min(x), max(x))

    axs[col].grid(True)
    axs[col].set_xlabel('date')
    axs[col].set_ylabel('percentage of mobility compared to day 0')

    axs[col].legend()

  axs[0].set_title('Mobility for a very restrictive lockdown')
  axs[1].set_title('Mobility for a restrictive lockdown')
  axs[2].set_title('Mobility for an unrestrictive lockdown')
  plt.legend()
  plt.tight_layout()
  plt.show()

def applemean(mobcountry_walking, mobcountry_transit, mobcountry_driving, country, interventions):

  if country == 'Korea':
    walking = mobcountry_walking[mobcountry_walking['region'].eq('Republic of Korea')]
    driving = mobcountry_driving[mobcountry_driving['region'].eq('Republic of Korea')]
    transiting = mobcountry_transit[mobcountry_transit['region'].eq('Republic of Korea')]
  else:
    if country == 'Catalonia':
      walking = mobcountry_walking[mobcountry_walking['region'].eq('Barcelona')]
      driving = mobcountry_driving[mobcountry_driving['region'].eq('Barcelona')]
      transiting = mobcountry_transit[mobcountry_transit['region'].eq('Barcelona')]
    else:
      walking = mobcountry_walking[mobcountry_walking['region'].eq(country)]
      driving = mobcountry_driving[mobcountry_driving['region'].eq(country)]
      transiting = mobcountry_transit[mobcountry_transit['region'].eq(country)]

  df = pd.concat([walking, driving, transiting])
  df_mean = df.drop(columns=['geo_type', 'region', 'transportation_type']).mean(axis=0)

  return df, df_mean

def plotmobapple(applemob, df_code, interventions):

  mobcountry_walking = applemob[applemob['transportation_type'] == 'walking']
  mobcountry_transit = applemob[applemob['transportation_type'] == 'transit']
  mobcountry_driving = applemob[applemob['transportation_type'] == 'driving']

  fig, axs = plt.subplots(len(df_code)//2, 2, sharex = True, sharey = True, figsize=(20, 20))

  for i, country in enumerate(df_code.index):

    df, df_mean = applemean(mobcountry_walking, mobcountry_transit, mobcountry_driving, country, interventions)
    c = df_code.iloc[i]['lang']

    mobility_g = interventions.loc[c]['Mobility']
    lockdown_g = interventions.loc[c]['Lockdown']
    normalcy_g = interventions.loc[c]['Normalcy']

    position = df.columns.get_loc(mobility_g)
    #position_2 = df.columns.get_loc(lockdown_g)
    #position_1 = df.columns.get_loc(normalcy_g)

    # Plot the dataframes on the same plot
    df.iloc[0, 3:].plot(ax=axs[i//2, i%2])
    df.iloc[1, 3:].plot(ax=axs[i//2, i%2])
    if country not in ['Korea', 'Serbia']:
      df.iloc[2, 3:].plot(ax=axs[i//2, i%2])
    df_mean.plot(ax=axs[i//2, i%2])

    axs[i//2, i%2].grid(True)
    axs[i//2, i%2].set_title('Mobility in ' + country)
    axs[i//2, i%2].set_xlabel('date')
    axs[i//2, i%2].set_ylabel('percentage of mobility compared to day 0')
    if country not in ['Korea', 'Serbia']:
      axs[i//2, i%2].legend(['Walking', 'Driving', 'Transit', 'Mean'])
    else:
      axs[i//2, i%2].legend(['Walking', 'Driving', 'Mean'])

    #axs[i//2, i%2].axvline(position_1-3, color='black', lw=2.2, linestyle="--")
    axs[i//2, i%2].axvline(position-3, color='blue', lw=1.5, linestyle="-", alpha=0.7)
    #axs[i//2, i%2].axvline(position_2-3, color='black', lw=1.5, linestyle="-", alpha=0.5)

  # Show the plot
  plt.tight_layout()
  plt.show()

def pageviewsplot(df_code, interventions, djson):
  fig, axs = plt.subplots(len(df_code)//2, 2, sharex=True, figsize=(20, 20))

  for i, c in enumerate(df_code['lang']):
      dt = djson[c]["topics"]["Culture.Media.Video games"]["percent"]

      mobility_g = interventions.loc[c]['Mobility']
      format_string = "%Y-%m-%d"

      # Convert the string to a numpy.datetime64 object
      date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

      dates = pd.to_datetime(list(dt.keys()))
      numbers = list(dt.values())

      if c == 'sv':
          x = [datetime.timestamp(k) for k in dates]
          x = x[365:]
          y = [val for val in numbers if not math.isnan(val)]
      else:
          x = [datetime.timestamp(k) for k in dates]
          y = numbers

      degree = 5
      coefficients = np.polyfit(x, y, degree)
      polynomial = np.poly1d(coefficients)

      y_fit = polynomial(x)

      row = i // 2
      col = i % 2

      axs[row, col].plot(dates, numbers)
      axs[row, col].plot(pd.to_datetime(x, unit='s'), y_fit)  # Convert x back to datetime for plotting

      axs[row, col].grid(True)

      axs[row, col].axvline(date_object, color='blue', lw=1.5, linestyle="-", alpha=0.7)
      axs[row, col].set_title('Percentage of Wikipedia page views related to video games in ' + df_code.index[i])
      axs[row, col].set_xlabel('Date')
      axs[row, col].set_ylabel('Percentage')
      axs[row, col].set_xlim(min(dates), max(dates))

      # Adjust x-axis labels
      # Get the dates for every 90 days
      selected_dates = pd.date_range(start=dates[0], end=dates[-1], freq='90D')

      # Format the dates as 'YYYY-MM-DD' and remove the time
      axs[row, col].set_xticks(selected_dates, selected_dates.strftime('%Y-%m-%d'), rotation=45)

  plt.tight_layout()
  plt.show()

def aggregatedpageviewsplot(df_code, djson, interventions):
  for i, c in enumerate(df_code['lang']):

      dt = djson[c]["topics"]["Culture.Media.Video games"]["percent"]
      dates = list(dt.keys())
      numbers = list(dt.values())
      dates = pd.to_datetime(dates)

      if c == 'sv':
          x = [datetime.timestamp(k) for k in dates]
          x = x[365:]
          y = [val for val in numbers if not math.isnan(val)]
      else:
          x = [datetime.timestamp(k) for k in dates]
          y = numbers

      #creating the approximated curve
      degree = 5
      coefficients = np.polyfit(x, y, degree)
      polynomial = np.poly1d(coefficients)
      y_fit = polynomial(x)

      #converting the mobility date (str) into a np.datetime64 to be able to use it
      mobility_g = interventions.loc[c]['Mobility']
      format_string = "%Y-%m-%d"
      date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

      index = dates.get_loc(date_object)
      mean = y_fit[0:index].mean()
      offset = 0 - mean
      y_fit = y_fit + offset

      plt.plot(pd.to_datetime(x, unit='s'), y_fit, label=c)  # Convert x back to datetime for plotting

      plt.grid(True)
      plt.title('Percentage of Wikipedia page views related to video games in depending on the country')
      plt.xlabel('Date')
      plt.ylabel('Percentage')
      plt.xlim(min(dates), max(dates))

      # Get the dates for every 90 days
      selected_dates = pd.date_range(start=dates[0], end=dates[-1], freq='90D')

      # Format the dates as 'YYYY-MM-DD' and remove the time
      plt.xticks(selected_dates, selected_dates.strftime('%Y-%m-%d'), rotation=45)

  plt.legend()
  plt.tight_layout()
  plt.show()

def desaggregatedpageviewsplot(df_code, djson, interventions):
  fig, axs = plt.subplots(1, 3, sharex = True, sharey = True, figsize=(18,6))

  for i, c in enumerate(df_code['lang']):

      dt = djson[c]["topics"]["Culture.Media.Video games"]["percent"]
      dates = list(dt.keys())
      numbers = list(dt.values())
      dates = pd.to_datetime(dates)

      if c == 'sv':
          x = [datetime.timestamp(k) for k in dates]
          x = x[365:]
          y = [val for val in numbers if not math.isnan(val)]
      else:
          x = [datetime.timestamp(k) for k in dates]
          y = numbers

      #creating the approximated curve
      degree = 5
      coefficients = np.polyfit(x, y, degree)
      polynomial = np.poly1d(coefficients)
      y_fit = polynomial(x)

      #converting the mobility date (str) into a np.datetime64 to be able to use it
      mobility_g = interventions.loc[c]['Mobility']
      format_string = "%Y-%m-%d"
      date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

      # Offset each curve
      index = dates.get_loc(date_object)
      mean = y_fit[0:index].mean()
      offset = 0 - mean
      y_fit = y_fit + offset

      col=0
      if c in ['fr', 'ca', 'it', 'sr']:
        row = 0
      else:
        if c in ['ko', 'ja', 'sv']:
          row = 2
        else:
          row = 1

      axs[row].plot(pd.to_datetime(x, unit='s'), y_fit, label=c)  # Convert x back to datetime for plotting

      axs[row].grid(True)
      axs[row].set_xlabel('Date')
      axs[row].set_ylabel('Percentage')
      axs[row].set_xlim(min(dates), max(dates))

      # Get the dates for every 90 days
      selected_dates = pd.date_range(start=dates[0], end=dates[-1], freq='90D')

      # Format the dates as 'YYYY-MM-DD' and remove the time
      axs[row].set_xticks(selected_dates, selected_dates.strftime('%Y-%m-%d'), rotation=45)

      axs[row].legend()

  axs[0].set_title('Variation of video games related page views \n for a very restrictive lockdown')
  axs[1].set_title('Variation of video games related page views \n for a restrictive lockdown')
  axs[2].set_title('Variation of video games related page views \n for an unrestrictive lockdown')
  plt.tight_layout()
  plt.show()

def correlationanalysisplot(df_code, globalmob, djson, interventions):

  fig, axs = plt.subplots(len(df_code)//2, 2, sharex=True, sharey = True, figsize=(20, 20))
  mean = meanmob(df_code, globalmob)

  for i, c in enumerate(df_code['lang']):

    av = mean[df_code.index[i]]

    if c == 'sv':
      zeros_360 = pd.Series([0] *360)
      zeros_25 = pd.Series([0] *25)
      mean_big = pd.concat([zeros_360, av, zeros_25])
      mean_big = mean_big.reset_index(drop=True)
      percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"]).dropna()
    else:
      zeros_775 = pd.Series([0] *775)
      mean_big = pd.concat([zeros_775, av])
      mean_big = mean_big.reset_index(drop=True)
      mean_big = mean_big[:-25]
      percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"])

    dates = list(percent.keys())
    dates = pd.to_datetime(dates)

    mobility_g = interventions.loc[c]['Mobility']
    format_string = "%Y-%m-%d"

    index = dates.get_loc(mobility_g)

    row = i // 2
    col = i % 2

    backwards = sm.tsa.ccf(percent, mean_big, adjusted=True)[::-1]
    forwards = sm.tsa.ccf(mean_big, percent, adjusted=True)
    ccf_output = np.r_[backwards[:-1], forwards]
    ccf_output = ccf_output[(len(ccf_output)//2)-50:(len(ccf_output)//2)+50]
    axs[row, col].stem(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, markerfmt='.')

    # Fill the space between the curve and the zero line with color
    axs[row, col].fill_between(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, 0, where=(ccf_output >= 0), interpolate=True, color='green', alpha=0.3, label='Positive Correlation')
    axs[row, col].fill_between(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, 0, where=(ccf_output < 0), interpolate=True, color='red', alpha=0.3, label='Negative Correlation')

    axs[row, col].grid(True)
    axs[row, col].set_xlabel('Lag (Days)')
    axs[row, col].set_ylabel('Cross-Correlation')
    axs[row, col].set_title('Cross-Correlation between Lockdown Intensity and Video Game Page Views in ' + df_code.index[i])
    axs[row, col].legend()

  plt.tight_layout()
  plt.show()

def correlationanalysisplot_false(df_code, globalmob, djson):

  fig, axs = plt.subplots(len(df_code)//2, 2, sharex=True, figsize=(20, 20))
  mean = meanmob(df_code, globalmob)

  for i, c in enumerate(df_code['lang']):

    row = i // 2
    col = i % 2

    av = mean[df_code.index[i]]

    percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"])
    p = {key[:10]: value for key, value in percent.items()}
    position = list(p.keys()).index('2020-02-15')

    backwards = sm.tsa.ccf(percent[position:], av[:-25], adjusted=True)[::-1]
    forwards = sm.tsa.ccf(av[:-25], percent[position:], adjusted=True)
    ccf_output = np.r_[backwards[:-1], forwards]
    ccf_output = ccf_output[(len(ccf_output)//2)-50:(len(ccf_output)//2)+50]
    axs[row, col].stem(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, markerfmt='.')

    # Fill the space between the curve and the zero line with color
    axs[row, col].fill_between(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, 0, where=(ccf_output >= 0), interpolate=True, color='green', alpha=0.3, label='Positive Correlation')
    axs[row, col].fill_between(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, 0, where=(ccf_output < 0), interpolate=True, color='red', alpha=0.3, label='Negative Correlation')

    axs[row, col].grid(True)
    axs[row, col].set_xlabel('Lag (Days)')
    axs[row, col].set_ylabel('Cross-Correlation')
    axs[row, col].set_title('Cross-Correlation between Lockdown Intensity and Video Game Page Views in ' + df_code.index[i])
    axs[row, col].legend()

  plt.tight_layout()
  plt.show()

def pearsonpageviews_false(df_code, globalmob, djson):

  mean = meanmob(df_code, globalmob)
  for i, c in enumerate(df_code['lang']):

    country_code = df_code.iloc[i]['state']
    av = mean[df_code.index[i]]
    country,_ = defglobalmob(country_code, globalmob)

    percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"])

    correlation_coefficient, p_value = pearsonr(av[:-25], percent[775:])
    print(f"For {c}, the correlation coefficient is: {correlation_coefficient}, and the p-value is: {p_value}")

def pearsonpageviews_plot(df_code, globalmob, djson):

  mean = meanmob(df_code, globalmob)

  # Lists to store correlation coefficients and corresponding p-values
  correlation_coefficients = []
  p_values = []

  for i, c in enumerate(df_code['lang']):
      country_code = df_code.iloc[i]['state']
      av = mean[df_code.index[i]]
      country, _ = defglobalmob(country_code, globalmob)

      percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"])

      # Calculate correlation coefficient and p-value
      correlation_coefficient, p_value = pearsonr(av[:-25], percent[775:])

      correlation_coefficients.append(correlation_coefficient)
      p_values.append(p_value)

  # Create a DataFrame for easy plotting with Seaborn
  df_plot = pd.DataFrame({'Country': df_code.index, 'Correlation Coefficient': correlation_coefficients, 'P-value': p_values})

  # Set the style of seaborn
  sns.set(style="whitegrid")

  # Create a bar plot
  plt.figure(figsize=(10, 6))
  ax = sns.barplot(x='Country', y='Correlation Coefficient', data=df_plot, palette=['red' if p > 0.05 else 'green' for p in p_values])
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

  # Add labels and title
  plt.xlabel('Country')
  plt.ylabel('Correlation Coefficient')
  plt.title('Correlation Coefficients between the moblity in the country and the pageviews')
  legend_handles = [
    Line2D([0], [0], marker='s', color='w', label='p-value > 0.05', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='p-value < 0.05', markerfacecolor='green', markersize=10)
    ]
  plt.legend(handles=legend_handles, loc='upper left')

  plt.show()

def pearsonpageviews_plot_inter(df_code, globalmob, djson):
    mean = meanmob(df_code, globalmob)

    correlation_coefficients = []
    p_values = []

    for i, c in enumerate(df_code['lang']):
        country_code = df_code.iloc[i]['state']
        av = mean[df_code.index[i]]
        country, _ = defglobalmob(country_code, globalmob)

        percent = pd.Series(djson[c]["topics"]["Culture.Media.Video games"]["percent"])

        correlation_coefficient, p_value = pearsonr(av[:-25], percent[775:])

        correlation_coefficients.append(correlation_coefficient)
        p_values.append(p_value)

    df_plot = pd.DataFrame({'Country': df_code.index, 'Correlation Coefficient': correlation_coefficients, 'P-value': p_values})

    # Create traces for each bar
    traces = []
    for i, country in enumerate(df_plot['Country']):

        if country in ['France', 'Catalonia', 'Italy', 'Serbia']:
          g = "groupe very restrictive"
        else:
          if country in ['Korea', 'Japan', 'Sweden']:
            g = "group unrestrictive"
          else:
            g = "group restrictive"

        trace = go.Bar(
            x=[country],
            y=[df_plot.loc[i, 'Correlation Coefficient']],
            name=country,
            marker=dict(color='red' if df_plot.loc[i, 'P-value'] > 0.05 else 'green'),
            legendgroup=g,
            legendgrouptitle_text=g,
            visible=True  # All traces are initially visible
        )
        traces.append(trace)

    layout = go.Layout(
        title='Correlation Coefficients between the mobility in the country and the pageviews',
        xaxis=dict(title='Country', range=[min(df_plot['Country']), max(df_plot['Country'])]),  # Set range for x-axis
        yaxis=dict(title='Correlation Coefficient', range=[min(df_plot['Correlation Coefficient'])-0.05, max(df_plot['Correlation Coefficient'])+0.05]),  # Set range for y-axis
        showlegend=True,
    )

    # Create frames for animation
    frames = [go.Frame(data=[trace], name=country) for country, trace in zip(df_plot['Country'], traces)]
    fig = go.Figure(data=traces, layout=layout, frames=frames)

    fig.show()

def medianpercentage_plot(df_code, globalmob, djson, interventions):

  data1 = {'Country': [],
          'percent': []}

  df_perc = pd.DataFrame(data1)

  for i, c in enumerate(df_code['lang']):

      cs = df_code.index[i]
      df, mobility = defglobalmob(cs, globalmob)
      dt = djson[c]["topics"]["Culture.Media.Video games"]["percent"]

      mobility_g = interventions.loc[c]['Mobility']
      normalcy_g = interventions.loc[c]['Normalcy']
      format_string = "%Y-%m-%d"

      # Convert the string to a numpy.datetime64 object
      date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

      dates = list(dt.keys())
      numbers = list(dt.values())

      dates = pd.to_datetime(dates)

      if c == 'sv':
          x = [datetime.timestamp(k) for k in dates]
          x = x[365:]
          y = [val for val in numbers if not math.isnan(val)]
      else:
          x = [datetime.timestamp(k) for k in dates]
          y = numbers

      df['date'] = pd.to_datetime(df['date'], utc=None)
      p = {key[:10]: value for key, value in dt.items()}
      position = list(p.keys()).index('2020-02-15')
      position2 = list(p.keys()).index(mobility_g)
      position3 = list(p.keys()).index(normalcy_g)

      median1 = np.nanmedian(numbers[position-30:position2])
      median2 =  np.nanmedian(numbers[position2:position3])

      percentage = (median2-median1)*100/median1

      new_row = pd.DataFrame({'Country': [cs], 'percent': [percentage], 'median_after_lockdown': [median2]})
      df_perc = pd.concat([df_perc, new_row], ignore_index=True)

  def get_color(country_code):
      if country_code in ['France', 'Catalonia', 'Italy', 'Serbia']:
          return 'tab:red'
      elif country_code in ['Korea', 'Japan', 'Sweden']:
          return 'yellow'
      else:
          return 'orange'

  # Apply the color function to create a 'color' column in the DataFrame
  df_perc['color'] = df_perc['Country'].apply(get_color)

  plt.figure(figsize=(10, 7))
  ax = sns.barplot(x='Country', y='percent', data=df_perc, palette=df_perc['color'])

  # Setting grid lines, title, and labels
  plt.grid(True)
  plt.title('Change in the proportions of pageviews related to \n video games amongst all the pageviews (before and after lockdown)')
  plt.xlabel('Countries')
  plt.ylabel('Percentage change')
  plt.xticks(rotation=45)
  plt.axhline(0, color='black', lw=1.5, linestyle="-", alpha=0.7)

  legend_handles = [
  Line2D([0], [0], marker='s', color='w', label='countries with very restrictive lockdown', markerfacecolor='tab:red', markersize=10),
  Line2D([0], [0], marker='s', color='w', label='countries with restrictive lockdown', markerfacecolor='orange', markersize=10),
  Line2D([0], [0], marker='s', color='w', label='countries with unrestrictive lockdown', markerfacecolor='yellow', markersize=10),
  ]
  plt.legend(handles=legend_handles, loc='upper right')

  # Show the plot
  plt.show()

def medianpercentage_boxplot(df_code, globalmob, djson, interventions):

  data1 = {'Country': [],
        'percent': [],
        'color': []}

  for i, c in enumerate(df_code['lang']):

      cs = df_code.index[i]
      df, mobility = defglobalmob(cs, globalmob)
      dt = djson[c]["topics"]["Culture.Media.Video games"]["percent"]

      mobility_g = interventions.loc[c]['Mobility']
      normalcy_g = interventions.loc[c]['Normalcy']
      format_string = "%Y-%m-%d"

      # Convert the string to a numpy.datetime64 object
      date_object = np.datetime64(datetime.strptime(mobility_g, format_string).date())

      dates = list(dt.keys())
      numbers = list(dt.values())

      dates = pd.to_datetime(dates)

      if c == 'sv':
          x = [datetime.timestamp(k) for k in dates]
          x = x[365:]
          y = [val for val in numbers if not math.isnan(val)]
      else:
          x = [datetime.timestamp(k) for k in dates]
          y = numbers

      df['date'] = pd.to_datetime(df['date'], utc=None)
      p = {key[:10]: value for key, value in dt.items()}
      position = list(p.keys()).index('2020-02-15')
      position2 = list(p.keys()).index(mobility_g)
      position3 = list(p.keys()).index(normalcy_g)

      median1 = np.nanmedian(numbers[position-30:position2])
      median2 =  np.nanmedian(numbers[position2:position3])

      percentage = (median2-median1)*100/median1

      def get_color(country_code):
        if country_code in ['France', 'Catalonia', 'Italy', 'Serbia']:
            return 'very restrictive'
        elif country_code in ['Korea', 'Japan', 'Sweden']:
            return 'unrestricitive'
        else:
            return 'restrictive'

      color = get_color(cs)

      data1['Country'].append(cs)
      data1['percent'].append(percentage)
      data1['color'].append(color)

  # Create DataFrame
  df_perc = pd.DataFrame(data1)

  fig = px.box(df_perc, x='color', y='percent', color = 'color', points='all', notched = True,
                  labels={'Correlation Coefficient': 'Correlation Coefficient'},
                  title='Correlation Coefficients between the mobility in the country and the pageviews')

  # Setting layout options
  fig.update_layout(
      title='Change in the proportions of pageviews related to video games (before and after lockdown)',
      xaxis=dict(title='Countries'),
      yaxis=dict(title='Percentage Change'),
      showlegend=True,
      legend=dict(title='Lockdown Category')
  )

  # Show the plot
  fig.show()

  # Output html that you can copy paste
  fig.to_html(full_html=False, include_plotlyjs='cdn')
  # Saves a html doc that you can copy paste
  fig.write_html("boxplot.html", full_html=False, include_plotlyjs='cdn')
