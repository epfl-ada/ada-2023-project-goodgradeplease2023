import pandas as pd
import numpy as np
import os
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
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.signal import savgol_filter
import json

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

def visualize_genres_distribution(stats_df,others_threadshold):
    df_others = stats_df[stats_df['Game'] <= others_threadshold]
    others_row = pd.DataFrame({'Main Genre': ['Others'], 'Game': [df_others['Game'].sum()]})
    stats_df = pd.concat([stats_df[stats_df['Game'] > others_threadshold], others_row], ignore_index=True)
    sns.set(style="whitegrid")

    # Plot the Barplot
    plt.figure(figsize=(10, 10))
    sns.set_color_codes("pastel")
    ax = sns.barplot(x="Game", y="Main Genre", data=stats_df, color="b")

    # Label every data
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.0f}', (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=12, color='black')

    plt.title('Game Genres Distribution')
    plt.show()


def visualize_pageviews_in_genre(pageviews_filepath, genres_filepath):
    pageviews = pd.read_csv(pageviews_filepath)
    game_genres = pd.read_csv(genres_filepath)
    pageviews.columns = ['Game','lang','timestamp','views']
    merged_df = pd.merge(pageviews, game_genres,on='Game',how='left')
    merged_df.dropna(inplace=True)
    grouped_df = merged_df.groupby(by=['Main Genre','timestamp','lang'],as_index=False).agg(pageviews = pd.NamedAgg(column='views',aggfunc='sum'))

    # We visualize the total pageviews according to the game genres on some main languages except English 
    fig, axes = plt.subplots(nrows=9, ncols= 3,figsize=(20,20))
    main_genres = list(set(grouped_df['Main Genre']))
    main_genres.remove('Comics')
    for index, ax in enumerate(axes.flat):
        genre = main_genres[index]
        sub_grouped_df = grouped_df[(grouped_df['Main Genre']==genre)&(grouped_df['lang'].isin(['de','fr','it','pt','es','ja']))]
        sub_grouped_df = sub_grouped_df.copy()
        sub_grouped_df['timestamp'] = pd.to_datetime(sub_grouped_df['timestamp'])
        for lang in sub_grouped_df['lang'].unique():
            lang_data = sub_grouped_df[sub_grouped_df['lang']==lang]
            ax.plot(lang_data['timestamp'],lang_data['pageviews'],label=lang)

        ax.set_title(genre)
        ax.legend()

    plt.tight_layout()
    plt.show()

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
    # Plot the pageviews and the mobility change in different countries.
    fig, axs = plt.subplots((len(df_code['lang']))//2, 2, sharey=True, figsize=(20, 20))

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

        row = i // 2
        col = i % 2

        mean_line, = axs[row, col].plot(df['date'], mean_g, label='Average change in mobility')
        for column in columns:
            if column == 'change from baseline':
                pageview_line, = axs[row, col].plot(df['date'], df[column], label='Change in pageviews in Games', color='red')
            else:
                axs[row, col].plot(df['date'], df[column], label=column, color='black', alpha=0.1)

        axs[row, col].axvline(lockdown_g, color='black', lw=2.2, linestyle="--")
        axs[row, col].axvline(mobility_g, color='blue', lw=1.5, linestyle="-", alpha=0.7)
        axs[row, col].axvline(normalcy_g, color='black', lw=1.5, linestyle="-", alpha=0.5)

        axs[row, col].set_xticks([13, 42, 73, 103, 134, 164], ['28 Feb', '28 Mar', '28 Apr', '28 May', '28 Jun', '28 Jul'])
        axs[row, col].set_xlim(min(df['date']), max(df['date']))
        axs[row, col].grid(True)
        axs[row, col].set_title(df_code.index[i])
        axs[row, col].set_xlabel('date')
        axs[row, col].set_ylabel('percentage of mobility compared to day 0(%)')
        axs[row, col].legend(handles=[mean_line, pageview_line], loc='upper right')
    plt.suptitle("Change of mobility and game pageviews in different countries in Year 2020",fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1,0.99])
    plt.show()

def visualize_different_language(lang,baseline,interventions, grouped_df,globalmob,code_dict,country_name, omit_genre=[]):
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