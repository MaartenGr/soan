import datetime

import numpy    as np

import matplotlib.pyplot      as plt
import matplotlib.dates       as mdates

from operator                 import gt, lt
from matplotlib.lines         import Line2D
from matplotlib.offsetbox     import OffsetImage, AnnotationBbox

def print_title(string, nr_tabs=0, symbol='#', head=True):
    """ Prints a title for the stats to be shown
    
    Example : header=True
    ##########
    ## test ##
    ##########
    
    Example : header=False
       test
    ##########
    
    
    Parameters:
    -----------
    string : str
        Title to be printed
    nr_tabs : int, default 0
        The number of tabs to be used to indent the title
    symbol : str, default '#'
        The symbol to be used as an outline
    head : boolean, default True
        A header is more clearly a header
    
    """
    length = len(string)
    
    nr_tabs = '\t' * nr_tabs
    
    if head:
        print(nr_tabs + symbol * length + symbol * 6)
        print(nr_tabs + symbol * 2 + ' ' + string + ' ' + 2 * symbol)
        print(nr_tabs + symbol * length + symbol * 6)
    else:
        print(nr_tabs + 3 * ' ' + string + ' ' * 3)
        print(nr_tabs + symbol * length + symbol * 6)
        
def print_avg_sentiment(df):
    """ Prints the average sentiment per user
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe with raw messages including a column called 
        'Sentiment' that has a value between -1 and 1
        
    """
    
    # Prints the average sentiment per user
    print_title('Average Sentiment', 3)
    for user in df.User.unique():
        avg_sentiment = round(np.mean(df[df.User == user]['Sentiment']), 3)
        print('{0: <30}'.format(user + ':') + '\t\t' + str(avg_sentiment))
    print('\n\n')

def print_sentiment(df):
    """ Prints 5 random positive and negative messages
    
    It returns positive messages with a sentiment higher
    than 0.8 and negative messages with sentiment lower
    than -0.3. Sentiment values are between -1 and 1. 
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe of all messages per user including
        a column called 'Sentiment' that includes values
        ranging from -1 to 1. 
    
    """
    
    # Prints 5 random positive and negative messages
    # Many are labelled with a 1 or -1, so getting the most
    # neg or pos messages will be too much
    for sentiment, oper in {'Positive': gt, 'Negative': lt}.items():
        print_title(sentiment + " Messages", 4)

        for user in df.User.unique():
            print_title(user+' Messages', 0, '-', head=True)


            if sentiment == 'Positive':
                temp = df[(df['Sentiment'] > 0.8) & (df.Message_Clean.str.len() > 10) & 
                            (df.User == user)]
            else:
                temp = df[(df['Sentiment'] < -0.3) & (df.Message_Clean.str.len() > 10) & 
                            (df.User == user)]
                
            word_list = []
            for index, row in temp.iterrows():
                word_list.append(row.Message_Clean)

            for _ in range(5):
                random_value = np.random.randint(0, len(word_list))
                print(word_list[random_value])
                del word_list[random_value]
                print()
        print()
        
def plot_sentiment(df, colors=None, savefig=False):
    """ Plots the weekly average sentiment over 
    time per user
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe of all messages per user including
        a column called 'Sentiment' that includes values
        ranging from -1 to 1. 
    colors : list, default None
        List of colors that can be used instead of the
        standard colors which are used if set to None
    savefig : boolean, default False
        If True it will save the figure in the 
        working directory
    """
    if not colors:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10
    
    # Resample to a week by summing
    df = df.set_index('Date')
    users = {}
    
    for user in df.User.unique():
        users[user] = df[df.User == user]
        users[user] = users[user].resample('7D').mean().reset_index()

        # Fill in missing values by taking the average of previous/next
        users[user]['Sentiment']  = users[user]['Sentiment'].interpolate()

    # Create figure and plot lines
    fig, ax = plt.subplots()

    legend_elements = []

    for i, user in enumerate(users):
        ax.plot(users[user].Date, users[user].Sentiment, linewidth=2, color=colors[i])

        user = user.split(' ')[0]
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=4, label=user))

    # Remove axis
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Limit graph
    plt.ylim(ymin=-1, ymax=1)
    # plt.xlim(xmin=datetime.date(2016, 8, 15))

    # Add lines instead of axis
    plt.axhline(-1, color='black', xmax=1, lw=7)
    # plt.axvline(datetime.date(2016, 8, 15), color='black', lw=7)

    # Setting emojis as y-axis
    font = {'fontname':'DejaVu Sans', 'fontsize':22}
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['\U0001f62D','\U0001f610','\U0001f604'], **font)

    # Set ticks to display month and year
    monthyearFmt = mdates.DateFormatter('%B %Y')
    ax.xaxis.set_major_formatter(monthyearFmt)
    plt.xticks(rotation=40)

    # Create legend    
    font = {'fontname':'Comic Sans MS', 'fontsize':24}
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.9, 1), loc=2, borderaxespad=0.)
    ax.set_title('Positivity of Messages', **font)

    # Set size of graph
    fig.set_size_inches(13, 5)
    fig.tight_layout()
    
    if savefig:
        fig.savefig('results/sentiment.png', dpi=300)
