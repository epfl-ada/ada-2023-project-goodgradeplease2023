# Let's load necessary libraries and the datasets
# "pip install seaborn" if necessary

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import math
from scipy.signal import savgol_filter
import json
from datetime import datetime
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

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
