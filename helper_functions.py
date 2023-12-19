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
    if choice == "Semi-restrictive":
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
    if english == "yes":
        data['England'] = ['en', 'GB']

    df_code = pd.DataFrame(data)
    df_code = df_code.transpose()
    df_code.rename(columns = {0:'lang', 1:'state'}, inplace = True)
    return data, df_code


def function1(d, df_code, interventions):
    fig, ax = plt.subplots(figsize=(20, 20))
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
        ax.plot(dates, numbers, color='lightgrey', lw=1)
        ax.plot(pd.to_datetime(x, unit='s'), y_fit, label=f'{df_code.index[i]} - Trend Line')  # Convert x back to datetime for plotting

        ax.axvline(date_object, color='blue', lw=1.5, linestyle="-", alpha=0.7)

        # Add individual lines to the list
        all_lines.append(y_fit)

    # Pad shorter arrays with NaN values
    all_lines_padded = [np.pad(line, (0, max_length - len(line)), 'constant', constant_values=np.nan) for line in all_lines]

    # Calculate the average line for all countries
    average_line = np.nanmean(all_lines_padded, axis=0)

    # Plot the average line as a thick blue line
    #ax.plot(pd.to_datetime(x, unit='s'), average_line, label='Average Trend', color='red', lw=2)

    ax.grid(True)
    ax.set_title('Percentage of Wikipedia page views related to video games')
    ax.set_xlabel('Date')
    ax.set_ylabel('Percentage')
    ax.set_xlim(min(dates), max(dates))
    ax.set_ylim(0, 0.015)

    # Adjust x-axis labels
    # Get the dates for every 90 days
    selected_dates = pd.date_range(start=dates[0], end=dates[-1], freq='90D')

    # Format the dates as 'YYYY-MM-DD' and remove the time
    ax.set_xticks(selected_dates, selected_dates.strftime('%Y-%m-%d'), rotation=45)

    # Add legend
    ax.legend()

    plt.tight_layout()
    plt.show()
    return
