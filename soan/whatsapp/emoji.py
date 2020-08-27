import re
import regex 
import operator

import emoji                as emoji_package
import numpy                as np
import matplotlib.pyplot    as plt
import seaborn              as sns

from collections            import Counter

sns.reset_orig() # Importing seaborn changes matplotlib look

def count_emojis(df, non_unicode = False):
    """ Calculates how often emojis are used between users
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe containing raw messages of whatsapp users
    non_unicode : boolean, default False
        Whether to count the non-unicode emojis or not
        
    Returns:
    --------
    emoji_all : dictionary of Counters
        Indicating which emojis are often used by which user
    """
    
    emoji_all = {}

    # Count all "actual" emojis and not yet text smileys
    for user in df.User.unique():
        # Count all sets of emojis
        temp_user = df.Emoji[(df.User == user) & (df.Emoji_Count < 20)].value_counts().to_dict()
        emoji_all[user] = {}

        # Go over all set of emojis
        for emojis, count in temp_user.items():
            
            # Create a list of emojis
            emojis = regex.findall(r'\p{So}\p{Sk}*', emojis)

            # Loop over individual emojis
            for emoji_value in emojis:

                # Skip empty values
                if emoji_value != '':     
                    try:
                        emoji_all[user][emoji_value] += count
                    except:
                        emoji_all[user][emoji_value] = count

                        
    # Count non-unicode smileys
    if non_unicode:
        for user in df.User.unique():
            # Loop over
            for _, row in df[(df.User == user) & (df.Different_Emojis.str.len() > 0)].iterrows():
                for some_emoji in row.Different_Emojis:
                    if len(some_emoji) > 1:
                        try:
                            emoji_all[user][some_emoji] += 1
                        except:
                            emoji_all[user][some_emoji] = 1        

    return emoji_all

def get_unique_emojis(df, counts, list_of_words):
    """ Uses TF-IDF to calculate which emoji are unique to
    which user. According to the following principal:
    
    TFIDF = (t_user + 1) / (words_user + 1) * log(sum_messages / t_all)
    Unique_Emoji = TFIDF_user / (TFIDF_all - TFIDF_user)
    
    Thus, it compares the TFIDF of a single user with all other users
    except that single user. 
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe containing raw messages of whatsapp users
    counts : dictionary of Counters
        Indicating which emojis are often used by which user
    list_of_words : list
        List of words that basically represent the dictionary of 
        possible words in the document. 
        
    Returns:
    --------
    unique_dict : dictionary of Counters
        Indicating which emojis are most unique to which user
    
    """
    tf_idf_dict = {user: {} for user in df.User.unique()}
    unique_dict = {user: {} for user in df.User.unique()}
    
    # Calculate TF-IDF for all smileys in that date range that were used
    # TF_IDF = (t_user + 1) / (words_user + 1) * log(sum_messages / t_all)
    for user in df.User.unique():
        for word in list_of_words:
            
            # Not all users may have said this word
            try:
                t_user = counts[user][str(word)]
            except:
                t_user = 0
              
            words_user = len(df[df.User == user])
            sum_messages = len(df)
            
            # Calculate t_all by trying to add the counts together
            # could be that a user doesnt use a smiley
            t_all = 0
            for user_2 in df.User.unique():
                try:
                    t_all += counts[user_2][str(word)]
                except:
                    t_all += 0
            
            # Calculate tf_idf and add it to the records
            tf_idf = (t_user + 1) / (words_user + 1) * np.log(sum_messages / t_all)
            tf_idf_dict[user][word] = tf_idf
                 
    # Calculate Unique words based on tf_idf
    # word_uniqueness = tf_idf_user / (tf_idf_all - tf_idf_user)
    for user in df.User.unique():
        for word in list_of_words:
            tf_idf_user = tf_idf_dict[user][word]
            tf_idf_all = sum([tf_idf_dict[u][word] for u in df.User.unique()])
            unique_dict[user][word] = tf_idf_user / (tf_idf_all - tf_idf_user)
    
    return unique_dict


def extract_emojis(str):
    """ Used to extract emojis from a string using the emoji package
    """
    return ''.join(c for c in str if c in emoji_package.UNICODE_EMOJI)

def prepare_data(df):
    """ Prepares the data by extracting and 
    counting emojis per row (per message).
    
    New columns:
    * Emoji - List of emojis that are in the message
    * Emoji_count - Number of emojis used in a message
    * Different_Emojis - Number of unique emojis used in a message
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe containing raw messages of whatsapp users
    Returns:
    --------
    df : pandas dataframe
        Dataframe containing raw messages of whatsapp users with
        the added columns containing information about emoji use
    
    """
    
    # Extract unicode emojis per message and count them
    df['Emoji'] = df.apply(lambda row: extract_emojis(str(row.Message_Clean)), 
                                       axis = 1)
    df['Emoji_Count'] = df.apply(lambda row: len(regex.findall(r'\p{So}\p{Sk}*', 
                                                                           row.Emoji)), axis = 1)
    
    # Find non-unicode smileys
    eyes, noses, mouths = r":;8BX=", r"-~'^", r")(/\|DPp"
    pattern = "[%s][%s]?[%s]" % tuple(map(re.escape, [eyes, noses, mouths]))
    df['Different_Emojis'] = df.apply(lambda row: re.findall(pattern, str(row.Message_Clean)), 
                                                  axis=1)
    
    return df


def print_stats(unique_emoji, counts, save=False):
    """ Prints the top 3 unique and often used emojis
    per user. 
    
    Parameters:
    -----------
    unique_emoji : dictionary of Counters
        Indicating which emojis are unique to which user
    counts : dictionary of Counters
        Indicating which emojis are often used by which user
    
    """
    if save:
        file = open("results/emoji.txt", "a")
    else:
        file = None
    
    print("#############################", file=file)
    print("### Unique Emoji (TF-IDF) ###", file=file)
    print("#############################", file=file)
    print(file=file)
    
    for user in unique_emoji.keys():
        print(user, file=file)
        unique_emoji[user] = Counter(unique_emoji[user])
        for emoji, score in unique_emoji[user].most_common(3):
            print(emoji, score, file=file)
        print(file=file)

    print("#########################", file=file)
    print("### Most Common Emoji ###", file=file)
    print("#########################", file=file)
    print(file=file)
    
    for user in counts.keys():
        print(user, file=file)
        counts[user] = Counter(counts[user])
        for emoji, score in counts[user].most_common(3):
            print(emoji, score, file=file)
        print(file=file)


def plot_counts(counts, user, savefig=False):
    """ Plots the counts of emoji for a single user 
    
    Parameters:
    -----------
    counts : dictionary of Counters
        Indicating which emojis are often used by which user
    user : str
        Indicates for which user the plot needs to be shown
    """
    # Prepare data
    sorted_x = sorted(counts[user].items(), key=operator.itemgetter(1), reverse=True)
    x = [x[0] for x in sorted_x][:10]
    y = [y[1] for y in sorted_x][:10]

    # Plot figure
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    bars = ax.bar(x, y,fc='#90C3D4', ec='#90C3D4', linewidth=3, width=.8, zorder=11)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.xaxis.set_tick_params(labelsize=0)
    plt.xticks([])
    
    # Set labels
    ax.set_ylabel('Nr Words')
    plt.title("Most often used Emoji")
    from matplotlib.font_manager import FontProperties

    # Load Apple Color Emoji font
    for rect1, label in zip(bars, x):
        height = rect1.get_height()
        plt.annotate(
            label,
            (rect1.get_x() + rect1.get_width() / 2, height + 5),
            ha="center",
            va="bottom",
            fontsize=30
        )
    # Show figure in a nice format
    plt.tight_layout()
    if savefig:
        fig.savefig(f'results/emoji_{user}.png', format="PNG", dpi=100)
    else:
        plt.show()

    
def plot_corr_matrix(df, user, list_of_words, counts):
    """ Plots a correlation matrix for the most commonly
    used emoji for a single user. 
    
    Most commonly used emoji are, at most, the 15 most often
    used emoji for a single user. 
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe containing raw messages of whatsapp users with
        the added columns containing information about emoji use
    user : str
        Indicates for which user the plot needs to be shown    
    list_of_words : list
        List of words that basically represent the dictionary of 
        possible words in the document. 
    counts : dictionary of Counters
        Indicating which emojis are often used by which user
    
    """
    
    # Create a dataframe with as columns all emoji and rows are counts of emoji
    df = df[df.User == user].copy()

    for emoji_str in list_of_words:
        df[emoji_str] = df.apply(lambda x: x.Message_Clean.count(emoji_str), 1)

    df = df[list_of_words]

    # Get most common emoji for a single user
    total_counts = Counter(counts[user])
    most_common = [emoji for emoji, _ in total_counts.most_common()][:15]
    df = df[most_common]
    df = df.T.drop_duplicates().T
    
    # Plot Correlation Matrix
    sns.set(style="white")
    sns.set_style({"font.sans-serif": "DejaVu Sans"})

    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})