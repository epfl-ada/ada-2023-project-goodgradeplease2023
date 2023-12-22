











import plotly.express as px
from casual import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from causalimpact import CausalImpact
from helper import *






def get_metrics():
    metrics = [
        ('Difference_Strategy', 'Ratio_Strategy'),
        ('Difference_Action', 'Ratio_Action'),
        ('Difference_Adult', 'Ratio_Adult'),
        ('Difference_Miscellaneous', 'Ratio_Miscellaneous')
    ]
    return metrics


import pandas as pd

def prepare_dataframe_for_timeseries(df, timestamp_column='timestamp', date_format='%Y-%m-%d'):
    """
    Prepares a DataFrame for time series analysis by converting a specified timestamp column 
    to datetime, setting it as the DataFrame index, and ensuring the index is in 
    the correct datetime format.

    Parameters:
    df (pd.DataFrame): DataFrame to be processed.
    timestamp_column (str): Name of the column containing the timestamp.
    date_format (str): Format of the timestamp in the original DataFrame.

    Returns:
    pd.DataFrame: Processed DataFrame ready for time series analysis.
    """
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], format=date_format)
    df.set_index(timestamp_column, inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

# Usage example:
# df_wikiviews = prepare_dataframe_for_timeseries(df_wikiviews)



def plot_comparison_subplots(comparison_final, metrics):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
   
    num_rows = len(metrics)
    
    # Create a list of subplot titles based on the metrics provided.
    # Ensure the titles correspond to the correct 'Difference' or 'Ratio' metric.
    subplot_titles = []
    for metric_difference, metric_ratio in metrics:
        subplot_titles.append(metric_difference.replace('_', ' '))
        subplot_titles.append(metric_ratio.replace('Ratio_', 'Percentage '))
    
    # Create a subplot grid with a specified number of rows and 2 columns
    # and include the subplot titles.
    fig = make_subplots(rows=num_rows, cols=2, shared_xaxes=True,
                        vertical_spacing=0.1,  # Increase spacing to accommodate titles
                        subplot_titles=subplot_titles)

    # Add traces for difference and ratio in their respective columns
    for row, (metric_difference, metric_ratio) in enumerate(metrics, start=1):
        # Add a bar plot for the metric difference on the left column
        fig.add_trace(
        go.Bar(
            x=comparison_final['country'],
            y=comparison_final[metric_difference],
            name=metric_difference,  # Set this to the metric's name
            marker_color='green'
        ),
        row=row, col=1
    )
    
        # Add a bar plot for the metric ratio on the right column
        fig.add_trace(
        go.Bar(
            x=comparison_final['country'],
            y=comparison_final[metric_ratio],
            name=metric_ratio,  # Set this to the metric's name
            marker_color='grey'
        ),
        row=row, col=2
    )

        # Update y-axis titles
        fig.update_yaxes(title_text='Difference in Means', row=row, col=1)
        fig.update_yaxes(title_text='% of Change(%)', row=row, col=2)
        
    # Update x-axis titles for the last row only
    fig.update_xaxes(title_text='Country', row=num_rows, col=1)
    fig.update_xaxes(title_text='Country', row=num_rows, col=2)

    # Adjust the layout and show the plot
    fig.update_layout(
        height=200 * num_rows,  # Adjust height to fit titles
        width=800,
        title_text='Comparison of different topics by Country',
        showlegend=False
    )
    fig.show()
    fig.write_html("navie.html")





def get_causal_impact_data():
    
    causal_impact_data = {
        'Action': {
            'France_Before': 41.25, 'France_After': 31.21, 'France_Sig': 1,
            'Italy_Before': 54.85, 'Italy_After': 23.33, 'Italy_Sig': 1,
            'Japan_Before': 14.68, 'Japan_After': -3.67, 'Japan_Sig': 0,
            'Germany_Before': -7.95, 'Germany_After': -9.22, 'Germany_Sig': 0
        },
        'Adult': {
            'France_Before': 53.65, 'France_After': 43.93, 'France_Sig': 1,
            'Italy_Before': 41.68, 'Italy_After': 13.39, 'Italy_Sig': 1,
            'Japan_Before': 34.28, 'Japan_After': 15.26, 'Japan_Sig': 1,
            'Germany_Before': 17.62, 'Germany_After': 12.72, 'Germany_Sig': 0
        },
        'Strategy': {
            'France_Before': 26.64, 'France_After': 15.23, 'France_Sig': 1,
            'Italy_Before': 40.58, 'Italy_After': 10.51, 'Italy_Sig': 1,
            'Japan_Before': 45.28, 'Japan_After': 21.43, 'Japan_Sig': 1,
            'Germany_Before': 27.41, 'Germany_After': 23.76, 'Germany_Sig': 1
        },
        'Miscellaneous': {
            'France_Before': 43.46, 'France_After': 33.53, 'France_Sig': 1,
            'Italy_Before': 64.94, 'Italy_After': 29.97, 'Italy_Sig': 1,
            'Japan_Before': 12.64, 'Japan_After': -6.12, 'Japan_Sig': 0,
            'Germany_Before': 129.32, 'Germany_After': 124.63, 'Germany_Sig': 1
        }
    }
    return causal_impact_data













def plot_causal_impact_with_updated_legend(causal_impact_data):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    
    # Create subplots: 2 rows, 2 columns
    fig = make_subplots(rows=2, cols=2, subplot_titles=tuple(causal_impact_data.keys()))

    # Define subplot position mapping
    subplot_pos = [(1, 1), (1, 2), (2, 1), (2, 2)]

    # Create a bar plot for each category
    for i, (category, pos) in enumerate(zip(causal_impact_data, subplot_pos)):
        data = causal_impact_data[category]
        countries = ['France', 'Italy', 'Japan', 'Germany']
        before_values = [data[country + '_Before'] for country in countries]
        after_values = [data[country + '_After'] for country in countries]
        significance = [data[country + '_Sig'] for country in countries]

        # Creating bar colors based on significance
        colors_before = ['green' if sig else 'darkgrey' for sig in significance]
        colors_after = ['darkgreen' if sig else 'grey' for sig in significance]

        # Plotting the 'Before' data
        fig.add_trace(go.Bar(x=countries, y=before_values, name='Before Division With Significance', 
                             marker_color=colors_before, legendgroup='Before', showlegend=(i==0)), 
                      row=pos[0], col=pos[1])

        # Plotting the 'After' data
        fig.add_trace(go.Bar(x=countries, y=after_values, name='After Division With Significance', 
                             marker_color=colors_after, legendgroup='After', showlegend=(i==0)), 
                      row=pos[0], col=pos[1])

    # Update layout for the figure
    fig.update_layout(
        title_text="Causal Impact of COVID-19 on Game Categories（Grey plot means without significance）",
        height=500,
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            traceorder='grouped',
            orientation='h',
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update xaxis and yaxis titles
    for i in range(1, 5):
        fig['layout']['xaxis' + str(i)].title.text = 'Countries'
        fig['layout']['yaxis' + str(i)].title.text = 'Relative Impact (%)'

    # Show the plot
    fig.show()
    fig.write_html("casual.html")


import pandas as pd

def merge_and_group_data(pageviews, game_genres):
    """
    Renames columns in the pageviews DataFrame, merges it with the game_genres DataFrame, 
    and groups and aggregates the views.

    Parameters:
    pageviews (pd.DataFrame): DataFrame containing pageviews data.
    game_genres (pd.DataFrame): DataFrame containing game genres data.

    Returns:
    pd.DataFrame: A grouped and aggregated DataFrame based on Main Genre, timestamp, and language.
    """
    # Renaming columns
    pageviews.columns = ['Game', 'lang', 'timestamp', 'views']

    # Merging DataFrames
    merged_df = pd.merge(pageviews, game_genres, on='Game', how='left')
    merged_df.dropna(inplace=True)

    # Grouping and aggregating views
    grouped_views_df = merged_df.groupby(by=['Main Genre', 'timestamp', 'lang'], as_index=False).agg(pageviews=pd.NamedAgg(column='views', aggfunc='sum'))

    return grouped_views_df



def generate_dataset(pageviews, game_genres, interventions):
    import numpy as np
    import pandas as pd
    
    # Renaming columns and merging data
    pageviews.columns = ['Game', 'lang', 'timestamp', 'views']
    merged_df = pd.merge(pageviews, game_genres, on='Game', how='left')
    merged_df.dropna(inplace=True)

    # Grouping and aggregating views
    grouped_views_df = merged_df.groupby(by=['Main Genre', 'timestamp', 'lang'], as_index=False).agg(pageviews=pd.NamedAgg(column='views', aggfunc='sum'))

    # Country codes to include in the analysis
    countries = {
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
        'En':['En','en']
    }

    # Creating language to country map and merging dataframes
    lang_to_country_map = {lang: country for country, langs in countries.items() for lang in langs}
    interventions['Country'] = interventions['lang'].map(lang_to_country_map)
    merged_df_1 = pd.merge(grouped_views_df, interventions[['lang', 'Mobility', 'Normalcy']], on='lang', how='left')
    merged_df_1 = merged_df_1.dropna(subset=['Mobility', 'Normalcy'])

    # Adjusting 'Period' categorization
    merged_df_1['Period'] = np.select(
        [
            merged_df_1['timestamp'] < merged_df_1['Mobility'],
            (merged_df_1['timestamp'] >= merged_df_1['Mobility']) & (merged_df_1['timestamp'] < merged_df_1['Normalcy']),
            merged_df_1['timestamp'] >= merged_df_1['Normalcy']
        ],
        ['Pre-Lockdown', 'During-Lockdown', 'Post-Lockdown'],
        default='Unknown'
    )

    # Recalculating average pageviews
    avg_pageviews = merged_df_1.groupby(['Main Genre', 'lang', 'Period'])['pageviews'].mean().reset_index()

    # Pivot and calculate DiD
    pivot_avg_pageviews = avg_pageviews.pivot_table(index=['Main Genre', 'lang'], columns='Period', values='pageviews').reset_index()
    pivot_avg_pageviews.columns.name = None
    pivot_avg_pageviews['Difference'] = pivot_avg_pageviews['During-Lockdown'] - pivot_avg_pageviews['Pre-Lockdown']
    pivot_avg_pageviews['Ratio'] = (pivot_avg_pageviews['During-Lockdown'] / pivot_avg_pageviews['Pre-Lockdown'] - 1) * 100
    pivot_avg_pageviews = pivot_avg_pageviews.drop(columns=['Post-Lockdown'])

    # Filtering and merging data for different genres
    Strategy = pivot_avg_pageviews[pivot_avg_pageviews['Main Genre'] == 'Strategy']
    Action = pivot_avg_pageviews[pivot_avg_pageviews['Main Genre'] == 'Action']
    Adult = pivot_avg_pageviews[pivot_avg_pageviews['Main Genre'] == 'Adult']
    Miscellaneous = pivot_avg_pageviews[pivot_avg_pageviews['Main Genre'] == 'Miscellaneous']
    strategy_action_merged = Strategy.merge(Action, on='lang', suffixes=('_Strategy', '_Action'))
    merged_inter = Adult.merge(Miscellaneous, on='lang', suffixes=('_Adult', '_Miscellaneous'))
    comparison = strategy_action_merged.merge(merged_inter, on='lang')

    # Filtering out English language
    comparison_final = comparison[comparison['lang'] != 'en']

    # Replacing language codes with country names
    country_codes = {
        # ... existing country codes ...
        'fr': 'France',
        'da': 'Denmark',
        'de': 'Germany',
        'it': 'Italy',
        'nl': 'Netherlands',
        'no': 'Norway',
        'sr': 'Serbia',
        'sv': 'Sweden',
        'ko': 'Korea',
        'ca': 'Catalonia',
        'fi': 'Finland',
        'ja': 'Japan'
    }
    comparison_final['country'] = comparison_final['lang'].map(country_codes)

    return comparison_final


def prepare_data_for_causal_impact(grouped_views_df):
    # Convert timestamp to datetime and set as index
    def process_df(df):
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
        df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    # Filtering and pivoting data for 'Adult' and 'Action' genres
    filter_1 = grouped_views_df[
        (grouped_views_df['Main Genre'].isin(['Adult', 'Action'])) &
        (grouped_views_df['lang'].isin(['fr', 'it', 'ja', 'de']))
    ]
    group_casual_1 = process_df(
        filter_1.pivot_table(
            index=['timestamp'],
            columns=['Main Genre', 'lang'],
            values='pageviews'
        ).reset_index()
    )

    # Flatten the MultiIndex in columns for group_casual_1
    group_casual_1.columns = ['_'.join(col).strip() if col[1] else col[0] for col in group_casual_1.columns.values]

    # Filtering and pivoting data for 'Strategy' and 'Miscellaneous' genres
    filter_2 = grouped_views_df[
        (grouped_views_df['Main Genre'].isin(['Strategy', 'Miscellaneous'])) &
        (grouped_views_df['lang'].isin(['fr', 'it', 'ja', 'de']))
    ]
    group_casual_2 = process_df(
        filter_2.pivot_table(
            index=['timestamp'],
            columns=['Main Genre', 'lang'],
            values='pageviews'
        ).reset_index()
    )

    # Flatten the MultiIndex in columns for group_casual_2
    group_casual_2.columns = ['_'.join(col).strip() if col[1] else col[0] for col in group_casual_2.columns.values]

    return group_casual_1, group_casual_2


from causalimpact import CausalImpact
import matplotlib.pyplot as plt

def analyze_causal_impact_before(group_casual_1, pre_period, post_period):
    
    # Apply CausalImpact
    ci = CausalImpact(group_casual_1['Action_fr'], pre_period, post_period)

    # Print the summary of the analysis
    print("Causal Impact Analysis for 'Action_fr'")
    print(ci.summary())

    # Plot the results
    ci.plot(panels=['original'], figsize=(15, 4))
    plt.show()


    



def analyze_causal_impact_with_ratio(group_casual_1, df_wikiviews):
    """
    Analyzes and plots the causal impact for the 'Action_fr' column in the provided DataFrame,
    using the ratio of this column to the 'fr' column of another DataFrame.
    """
    # Initialize a dictionary to store the results
    impact_results = {}

    # Define the pre-intervention and post-intervention periods for France
    pre_period_fr = ['2019-10-16', '2020-03-14']
    post_period_fr = ['2020-03-15', '2020-07-02']

    # Calculate the percentage ratio
    df_percentage = group_casual_1['Action_fr'] / df_wikiviews['fr']

    # Apply CausalImpact
    ci = CausalImpact(df_percentage, pre_period_fr, post_period_fr)
    impact_results['Action_fr'] = ci.summary_data

    # Print the summary of the analysis and plot the results
    print('Causal Impact Analysis for \'Action_fr\' After division by total views')
    print(ci.summary())
    ci.plot(panels=['original'], figsize=(15, 4))
    plt.show()

    return impact_results



