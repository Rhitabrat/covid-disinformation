import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from skimage import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import statistics
from geopy.geocoders import Nominatim
import plotly.graph_objects as go
import plotly.figure_factory as ff

app = dash.Dash(__name__)

# layout color
colors = {
    'background': '#071734',
    'text': '#ffffff',
}


''' 
Data Exploration 
'''


### ------------ Exploring details of the data ------------###

# import the data
df = pd.read_csv("data/challenge_data_2.csv", delimiter=',', encoding='latin1')

# get the shape
df_shape = df.shape

# get random 10% rows
percent_of_data_used = 0.9
df_train, df_test = train_test_split(df, test_size=percent_of_data_used)

# df_train = df_train[df_train.notnull().all(axis = 1)]

# get the shape of df_train
df_train_shape = df_train.shape

# get the column names
df_train_columns = df_train.columns.values


### ------------ Creating a piechart for 'source' column ------------###
def createSourcePieChart(data):
    df = data.source

    # get the top 10 sources of tweet
    top_sources = pd.value_counts(df)[:10]

    # print(top_sources.values)
    fig = px.pie(top_sources, names=top_sources.index, values=top_sources.values,
                 color_discrete_sequence=px.colors.sequential.Aggrnyl,)

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )

    return fig, top_sources

### ------------ Creating an area chart for 'display_text_width' column ------------###
def createLengthAreaChart(data):
    # df = data.display_text_width

    fig = px.area(data, y=data.display_text_width,
                  labels={
                      "display_text_width": "Length of Tweet",
                      "index": "Tweet"
                  },
                  )

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    
    return fig

### ------------ Creating a word cloud for the tweets ------------###
def createWordcloud(data):

    data = data[data['text'].notnull()].copy()
    
    # remove hyperlinks
    data['text'] = data['text'].str.replace('http\S+|www.\S+', '', case=False)

    # remove stopwords
    stop = stopwords.words('english')
    data['text'].apply(lambda x: [item for item in x if item not in stop])

    # remove @username
    data['text'] = data['text'].replace('@[\w]+', '',regex=True)

    text = data["text"].values

    wordcloud = WordCloud(background_color=colors['background'], collocations=False).generate(str(text))

    figure = plt.figure()
    figure.set_facecolor(colors['background'])
    
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('wordcloud.png')
    
    img = io.imread('wordcloud.png')
    fig = px.imshow(img)
    
    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text'],
        yaxis_visible=False,
        yaxis_showticklabels=False,
        xaxis_visible=False,
        xaxis_showticklabels=False,   
    )
    
    
    return fig

### ------------ Creating a bar chart for 'account_created_at' column ------------###
def createDateBarChart(data):

    data = data[data['text'].notnull()].copy()
    
    data.account_created_at = pd.to_datetime(data['account_created_at']).dt.year
    
    count_by_year = pd.value_counts(data.account_created_at)
    fig = px.bar(data, x=count_by_year.index,y=count_by_year.values,labels={
                      "x": "Year",
                      "y": "Number of Accounts Created"
                  },)

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    
    return fig, count_by_year
    
### ------------ Creating a bar chart for 'verified' and 'location' column ------------###

# # convert locations into country names
# geolocator = Nominatim(user_agent = "Covid Disinformation") 

# def getCountry(address):
#     try:
#         location = geolocator.geocode(address)
#         return list(location)[0].split(', ')[-1]
#     except TypeError:
#         return address
    

def createVerifiedLocationBarChart(data):

    data = df_train[df_train['text'].notnull()].copy()

    data = data[data['location'].notna()]

    false_count_by_country = pd.value_counts(data['location'].loc[data['verified'] == False])
    df_false_count_by_country = pd.DataFrame({'location':false_count_by_country.index, 'count':false_count_by_country.values})
    df_false_count_by_country['verified'] = [False]*len(df_false_count_by_country)
    df_false_count_by_country = df_false_count_by_country[~df_false_count_by_country
                                                        ['count'].isin([1, 2])]

    true_count_by_country = pd.value_counts(data['location'].loc[data['verified'] == True])
    df_true_count_by_country = pd.DataFrame({'location':true_count_by_country.index, 'count':true_count_by_country.values})
    df_true_count_by_country['verified'] = [True]*len(df_true_count_by_country)
    # df_true_count_by_country = df_true_count_by_country[~df_true_count_by_country
                                                        # ['count'].isin([1, 2])]

    result = pd.concat([df_true_count_by_country, df_false_count_by_country])



    # for num in range(0, len(result.location), 30):
    #     result['location'][num:num+30] = result['location'][num:num+30].map(getCountry)

    fig = px.bar(result, x='location', y='count', color='verified', labels={
                        "x": "Country",
                        "y": "Number of tweet"
                    },)

    fig.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )

    return fig
    

### ------------ Creating a bar chart for emotion columns ------------###
def createEmotionLineChart(data):

    data = data[data['text'].notnull()].copy()
    
    # data.account_created_at = pd.to_datetime(data['account_created_at']).dt.month
    
    # data['account_created_at'] = pd.to_datetime(data.account_created_at)
    data = data.sort_values(by='account_created_at')
    
    # data = data[~data['anger'].isin([0])]
    
    # fig = px.line(data.head(), x='account_created_at', y='anger', labels={
    #                   "x": "Year",
    #                   "y": "Emotion"
    #               },)

    # fig.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    
    emotions = ['anger', 'anticipation', 'disgust',]
    
    fig = go.Figure(data=go.Heatmap(
        z=data[['anger', 'anticipation', 'disgust']],
        x=data.account_created_at,
        y=emotions,
        colorscale='Viridis'))
    
    print(data.account_created_at.head())
    
    return fig




''' 
Dash Part 
'''


app.layout = html.Div(
    style={
        'padding-left': '20%',
        'padding-right': '20%',
    },
    children=[
        html.H2(
            children='COVID Disinformation',
        ),
        html.Div(
            children='Dimension of the given data: '+str(df_shape)
        ),
        html.Br(),
        html.Div(
            children='Number of tweets in the given data set: ' +
            str(df_shape[0])
        ),
        html.Br(),
        html.Div(
            children='Percentage of random data extracted from the given data: ' +
            str(round((1-percent_of_data_used)*100))+" %"
        ),
        html.Br(),
        html.Div(
            children='Number of tweets after extraction: ' +
            str(df_train_shape[0])
        ),
        html.Br(),
        html.Div(
            children='Name of all the columns: '+str(df_train_columns)
        ),
        html.Br(),
        html.H3(
            children='Top 10 Sources of Tweet',
        ),
        dcc.Graph(
            id='pie-chart-source',
            figure=createSourcePieChart(df_train)[0]
        ),
        html.Br(),
        html.Div(
            children='Maximum information is spread from '+str(createSourcePieChart(df_train)[1].index[0])+'.'
        ),
        html.Br(),
        html.H3(
            children='Length of tweets',
        ),
        dcc.Graph(
            id='line-chart-display_text_width',
            figure=createLengthAreaChart(df_train)
        ),
        html.Br(),
        html.Div(
            children='Average length of characters of a tweet is '+str(round(statistics.mean(df_train.display_text_width)))+'.'
        ),
        html.Br(),
        html.H3(
            children='Word Cloud',
        ),
        dcc.Graph(
            id='wordcloud',
            figure=createWordcloud(df_train)
        ),
        html.Br(),
        html.H3(
            children='Dates of creation of the accounts',
        ),
        dcc.Graph(
            id='date',
            figure=createDateBarChart(df_train)[0]
        ),
        html.Br(),
        html.Div(
            children='Maximum number of accounts were created in the years (descending order): '+str(list(createDateBarChart(df_train)[1].index[:5]))+'.'
        ),
        html.Br(),
        html.H3(
            children='Number of verified users per location',
        ),
        html.Div(
            children='Note: Locations having 1 or 2 unverified accounts are removed. Duplicate names of the same location could have been merged to a single name. However, due to the lack of time and socket.timeout (bad internet connection), I was not able to do so.',
        ),
        dcc.Graph(
            id='country',
            figure=createVerifiedLocationBarChart(df_train)
        ),
        html.Br(),
        html.H3(
            children='Emotion development over time',
        ),
        dcc.Graph(
            id='emotion',
            figure=createEmotionLineChart(df_train)
        ),
    ])


'''
Driver code
'''


if __name__ == '__main__':
    app.run_server(debug=True)
