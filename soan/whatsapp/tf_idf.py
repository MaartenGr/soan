import requests

import pandas as pd
import numpy as np

from PIL import Image
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

import nltk
from nltk.corpus import stopwords as nltk_stopwords
nltk.download('stopwords')


def count_words_per_user(df, sentence_column = "Message_Only_Text", user_column = "User"):
    """ Creates a count vector for each user in which
        the occurence of each word is count over all 
        documents for that user. 

    Parameters:
    -----------
    df : pandas dataframe
        Dataframe of all messages
    sentence_column : string, default 'Message_Only_Text'
        Name of the column of which you want to 
        create a word count
    user_column : string, default 'User'
        Name of the column that specifies the user
        
    Returns:
    --------
    df : pandas dataframe
        Dataframe counts per word per user
    
    """
    # Creating a dataframe with all words
    counts = list(Counter(" ".join(list(df[sentence_column])).split(" ")).items())
    counts = [word[0] for word in counts]
    counts = pd.DataFrame(counts, columns = ['Word'])
    counts = counts.drop(0)

    # Adding counts of each user to the dataframe
    for user in df.User.unique():
        count_temp = list(Counter(" ".join(list(df.loc[df[user_column] == user, 
                                                       'Message_Only_Text'])).split(" ")).items())
        counts[user] = 0
        for word, count in count_temp:
            counts.loc[counts['Word'] == word, user] = count
            
    counts = counts[counts.Word.str.len() > 1]
            
    return counts


def remove_stopwords(df, language=False, path='', column = "Word"):
    """ Remove stopwords from a dataframe choosing
    a specific column in which to remove those words
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe of counts per word per user
    path : string, default ''
        Path of the file that contains the stopwords
    language : str, default False
        The language to be used in the built-in nltk stopwords
    column : string, default 'Word'
        Column to clean

    Returns:
    --------
    df : pandas dataframe
        Dataframe of counts per word per user
        excluding the stopwords
    
    """

    if language:
        try:
            stopwords = nltk_stopwords.words(language)
        except:
            languages = nltk_stopwords.fileids()
            raise Exception(f"Please select one of the following languages: {languages}")

    else:
        with open(path) as stopwords:
            stopwords = stopwords.readlines()
            stopwords = [word[:-1] for word in stopwords]

    df = df[~df[column].isin(stopwords)]
    
    return df

def get_unique_words(counts, df_raw, version):
    """ Get a list of unique words 
    
    The dataframe needs be structured as follows:
    First column is called "Word" and contains a certain word
    Any following columns are named as the users and contain the
    count of each word. 
    
    |   |    Word   | Tim | Nadia | 
    | 1 | pride     | 0   | 1     |
    | 2 | groceries | 2   | 9     |
    etc. 
    
    Formulas:
    t_user = Number of times word t said by user
    t_all = Number of times word t said by all users
    sum_messages = Number of all messages
    messages_user = Number of messages user has send
    sum_words = Number of all words
    words_user = Number of words user has send
    
    Version A
    TF_IDF = ((t_user+1)^2 / t_all) * (sum_messages / messages_user)

    Version B
    TF_IDF = ((t_user+1)^2 / t_all) * (sum_words / words_user)

    Version C
    TF_IDF = (t_user + 1) / (words_user + 1) * log(sum_messages / t_all)
    
    Parameters:
    -----------
    counts : pandas dataframe
        Dataframe of counts per word per user
    df_raw : pandas dataframe
        Dataframe of raw messages
    version : string
        Which formula to use (A, B, C)

    Returns:
    --------
    df_words : pandas dataframe
        Dataframe tf_idf scores per word per user and unique value
    
    """
    
    df_words = counts.copy()
    
    # Number of messages by i 
    nr_messages = {user: len(df_raw[df_raw.User == user]) for user in df_words.columns[1:]}
    nr_users = len(nr_messages.keys())
    nr_words = {user: np.sum(df_words[user]) for user in df_words.columns[1:]}
    total = sum(nr_messages.values())

    # Calculate TF_IDF based on the version
    for user in nr_messages.keys():
        df_words[user+"_TF_IDF"] = df_words.apply(lambda row: tf_idf(row, user, 
                                                                    nr_users, nr_words,
                                                                    nr_messages, version=version), 
                                              axis = 1)

    # TF_IDF divided by each other so we can see the relative importance
    for user in nr_messages.keys():
        df_words[user+"_Unique"] = df_words.apply(lambda row: word_uniqueness(row, 
                                                                             nr_users,
                                                                             user),
                                                  axis = 1)
        
    return df_words

def tf_idf(row, user, nr_users, nr_words, nr_messages, version):
    """ Used as a lambda function inside get_unique_words() to 
        get the tf_idf scores based on one of three formulas
    
    Formulas:
    t_user = Number of times word t said by user
    t_all = Number of times word t said by all users
    sum_messages = Number of all messages
    messages_user = Number of messages user has send
    sum_words = Number of all words
    words_user = Number of words user has send
    
    Version A
    TF_IDF = ((t_user+1)^2 / t_all) * (sum_messages / messages_user)

    Version B
    TF_IDF = ((t_user+1)^2 / t_all) * (sum_words / words_user)

    Version C
    TF_IDF = (t_user + 1) / (words_user + 1) * log(sum_messages / t_all)
    
    """
    
    # TF_IDF = (t_user^2 / t_all) * (sum of messages / messages by user)
    if version == "A":
        t_user = row[user]
        t_all =  np.sum(row.iloc[1:nr_users+1])
        sum_messages = sum(nr_messages.values())
        messages_user = nr_messages[user]
        
        tf_idf = (np.square(t_user + 1) / (t_all)) * (sum_messages / messages_user)
        
        return tf_idf
    
    # TF_IDF = (t_user^2 / t_all) * (sum of words / words by user)
    elif version == "B":
        t_user = row[user]
        t_all =  np.sum(row.iloc[1:nr_users+1])
        sum_words = sum(nr_words.values())
        words_user = nr_words[user]
        
        tf_idf = (np.square(t_user + 1) / (t_all)) * (sum_words / words_user)
        
        return tf_idf
    
    # TF_IDF = (t_user / words_user) * log(sum of messages / t_all)
    elif version == "C":
        t_user = row[user]
        words_user = nr_words[user]

        sum_messages = sum(nr_messages.values())
        t_all =  np.sum(row.iloc[1:nr_users+1])
        
        tf_idf = (t_user + 1 / words_user + 1) * np.log(sum_messages / t_all)
        
        return tf_idf
    
def word_uniqueness(row, nr_users, user):
    """ Used as a lambda function in function get_unique_words()
    
    Formula:
    
    word_uniqueness = tf_idf_user / (tf_idf_all - tf_idf_user)
    
    """
    
    tf_idf_user = row[user+"_TF_IDF"]
    tf_idf_all = np.sum(row.iloc[nr_users+1: 2*nr_users+1])
    
    with np.errstate(divide='ignore'):
        unique_value_user = np.divide(tf_idf_user, 
                                      (tf_idf_all - tf_idf_user))
    
    return unique_value_user


def plot_unique_words(df_unique, user, image_path=None, image_url=None, save=None,
                      title=" ", title_color="white", title_background="black", font=None, 
                      width=None, height=None):
    """
    
    Parameters:
    -----------
    df_unique : dataframe
        Dataframe containing a column "Word" and a column
        user+"_Unique" that describes how unique a word is
        by simply giving a floating value
    user : string 
        The name of the user which is the user in the column user+"_Unique"
    image_path : string with // to the path 
        Path to the picture you want to use
    image_url : string 
        Url to the image you want to use
    save : string
        If you want to save the name then simply set a name without extension
    title : string
        Title of the plot
    title_color : string
        Color of the title
    title_background : string
        Color of the background box of the title
    font : string
        Family font to use (make sure to check if you have it installed)
    width : integer or float
        Width of the plot (will also resize the image)
    height : integer or float
        Height of the plot (will also resize the image)
    """

    # Set font to be used
    if font:
        font = {'fontname':font}
    else:
        font = {'fontname':'Comic Sans MS'}

    # Background image to be used, black if nothing selected
    if image_path:
        img = mpimg.imread(image_path)
        img = Image.open(image_path)
    elif image_url:
        img = Image.open(requests.get(image_url, stream=True).raw)
    else:
        img = np.zeros([100,100,3],dtype=np.uint8)
        img.fill(0) 
    
    if width and height:
        img = img.resize((width, height))
    else:
        # Get size of image
        width = img.shape[1]
        height = img.shape[0]

    # Prepare data for plotting
    # to_plot = get_unique_words(counts, df_raw, version = 'C')
    to_plot = df_unique.sort_values(by=user+'_Unique', ascending=True)
    to_plot = to_plot.tail(10)[['Word', user+'_Unique']].copy()
    
    # Create left part of graph ('top') and right part which overlays
    # the image ('bottom')
    to_plot['top'] = (to_plot[user+'_Unique'] * (width*0.99) ) / max(to_plot[user+'_Unique']) 
    to_plot['bottom'] = width - to_plot['top'] 

    # Create the steps of the bars based on the height of the image
    steps = height/len(to_plot)
    y_pos = [(height/len(to_plot)/2) + (i * steps) for i in range(0, len(to_plot))]

    # Plot figure
    fig, ax = plt.subplots()

    # First plot the image
    plt.imshow(img, extent=[0, width*0.99, 0, height], zorder=1)

    # Then plot the right part which covers up the right part of the picture
    ax.barh(y_pos, to_plot['bottom'], left=to_plot['top'],height=steps, color='w',align='center',
            alpha=1,lw=2, edgecolor='w', zorder=2)

    # Finally plot the bar which is fully transparent aside from its edges
    ax.barh(y_pos, to_plot['top'], height=steps, fc=(1, 0, 0, 0.0), align='center',lw=2,
            edgecolor='white',zorder=3)

    # Remove ticks
    ax.yaxis.set_ticks_position('none') 
    ax.xaxis.set_ticks_position('none') 

    # Set labels and location y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(to_plot['Word'].values), fontsize=18,**font)
    ax.set_ylim(top=height)

    # Make them with to remove any image line that may be left
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')

    # Remove the left and bottom axis
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add a small patch that removes some of the extra background at the top
    ax.add_patch(patches.Rectangle((0,height),width, 20,facecolor='white',linewidth = 0, zorder=3))

    # Add left and bottom lines
    plt.axvline(0, color='black', ymax=1, lw=5, zorder=4)
    plt.axvline(width, color='white', ymax=1, lw=5, zorder=5)
    plt.axhline(0, color='black', xmax=1, lw=5, zorder=6)
    plt.axhline(height, color=title_background, xmax=1, lw=3, zorder=7)

    # Create Title Box
    # This might be a temporary solution as 
    # makes_axes_locatable might lose its functionality
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="9%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    at = AnchoredText(title, loc=10, pad=0,
                      prop=dict(backgroundcolor=title_background,
                                size=23, color=title_color, **font))
    cax.add_artist(at)
    cax.set_facecolor(title_background)   
    cax.spines['left'].set_visible(False)
    cax.spines['bottom'].set_visible(False)
    cax.spines['right'].set_visible(False)
    cax.spines['top'].set_visible(False)
                   
    fig.set_size_inches(10, 10)
    if save:
        plt.savefig(f'results/{save}_tfidf.png', dpi = 300)
        
def print_users(df):
    print("#" * (len('Users')+8))
    print("##  " + 'Users' + "  ##" )
    print("#" * (len('Users')+8))
    print()
    
    for user in df.User.unique():
        print(user)