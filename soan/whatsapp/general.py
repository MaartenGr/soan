import random
import datetime
import itertools
import calendar
import datetime

import pandas   as pd
import numpy    as np

import matplotlib.pyplot    as plt
import matplotlib.dates     as mdates

from matplotlib.colors      import ColorConverter, ListedColormap
from matplotlib.lines       import Line2D

def print_users(df):
    """ Prints the names of the users so that the exact
    name can be used for other functions. 
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe of all messages
    """
    
    print("#" * (len('Users')+8))
    print("##  " + 'Users' + "  ##" )
    print("#" * (len('Users')+8))
    print()
    
    for user in df.User.unique():
        print(user)

def plot_active_hours(df, color='#ffdfba', savefig=False, dpi=100, user='All'):
    """ Plot active hours of a single user or all 
    users in the group. A bar is shown if in that hour
    user(s) are active by the following standard:
    
    If there are at least 20% of the maximum hour of messages
    
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe of all messages
    color : str, default '#ffdfba'
        Hex color of bars
    savefig : boolean, deafult False
        Whether or not to save the figure instead of showing
    dpi : int, default 100
        Resolution of the figure you want to save
    user : str, default 'All'
        Variable to choose if you want to see the active hours
        of a single user or all of them together. 'All' shows 
        all users and the name of the user shows that user. 
    
    """
    # Prepare data for plotting
    if user != 'All':
        df = df.loc[df.User == user]
        title = 'Active hours of {}'.format(user)
    else:
        title = 'Active hours of all users'
        
    hours = df.Hour.value_counts().sort_index().index
    count = df.Hour.value_counts().sort_index().values
    font = {'fontname':'Comic Sans MS'}

    # Only get active hours
    count = [1 if x > (.2 * max(count)) else 0 for x in count]

    # Plot figure
    fig, ax = plt.subplots()
    
    # Then plot the right part which covers up the right part of the picture
    ax.bar(hours, count, color=color,align='center', width=1,
            alpha=1, lw=4, edgecolor='w', zorder=2)

    # Set ticks and labels
    ax.yaxis.set_ticks_position('none') 
    ax.set_yticks([])
    ax.set_ylabel('', labelpad=50, rotation='horizontal',
                   color="#6CA870",**font)
    ax.set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24])
    ax.set_xticklabels(["Midnight", "3 AM", "6 AM", "9 AM", "Noon", "3 PM", "6 PM", "9 PM", 
                       "Midnight"], **font)
    plt.title(title, y=0.8)
    
    # Create horizontal line instead of x axis
    plt.axhline(0, color='black', xmax=1, lw=2, zorder=3, clip_on=False)

    # Make axes white to remove any image line that may be left
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # Remove the left and bottom axis
    ax.spines['left'].set_visible(False)

    # Set sizes
    fig.set_size_inches((13.5, 1))
    fig.tight_layout(rect=[0, 0, .8, 1])

    # Save or show figure    
    if savefig:
        plt.savefig(f'results/{savefig}active_hours.png', dpi = dpi)
    else:
        plt.show()


def plot_active_days(df, savefig=False, dpi=100, user='All'):
    """ Plot active day of a single user or all 
    users in the group. The height of a bar indicates
    how active people are on that day. 
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe of all messages
    savefig : boolean, deafult False
        Whether or not to save the figure instead of showing
    dpi : int, default 100
        Resolution of the figure you want to save
    user : str, default 'All'
        Variable to choose if you want to see the active hours
        of a single user or all of them together. 'All' shows 
        all users and the name of the user shows that user. 
    """
    
    # Prepare data
    if user != 'All':
        df = df.loc[df.User == user, :]
        title = 'Active days of {}'.format(user)
    else:
        title = 'Active days of all users'

    day_of_week = df.apply(lambda row: row.Date.dayofweek, axis=1)
    days = day_of_week.value_counts().sort_index().index
    count = day_of_week.value_counts().sort_index().values

    # Plot figure
    fig, ax = plt.subplots()
    font = {'fontname':'Comic Sans MS'}

    # Then plot the right part which covers up the right part of the picture
    ax.bar(days, count, color='#90C3D4',align='center', width=0.9,
            alpha=1, lw=2, edgecolor='w', zorder=2)

    # # Remove ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')

    # # Set labels and location y-axis
    ax.set_yticks([])
    ax.set_xticks(days)
    ax.set_xticklabels(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                      **font)

    # # Make them with to remove any image line that may be left
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

    # # Remove the left and bottom axis
    ax.spines['left'].set_visible(False)

    fig.set_size_inches((10, 4))
    plt.title(title)
    
    # Save or show figure    
    if savefig:
        plt.tight_layout()
        plt.savefig(f'results/{savefig}_active_days.png', dpi = dpi)
    else:
        plt.show()

def plot_day_spider(df, colors=None, savefig=False, dpi=100):
    """ Plot active days in a spider plot with all users
    shown seperately. 
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe containing all messages
    colors : list, default None
        List of colors to be used for the plot. 
        Random colors are chosen if nothing is chosen
    savefig : boolean, deafult False
        Whether or not to save the figure instead of showing
    dpi : int, default 100
        Resolution of the figure you want to save
        
    """
    
    # Initialize colors
    if not colors:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10

    # Get count per day of the week
    categories = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    N = len(categories)
    count = list(df.Day_of_Week.value_counts().sort_index().values)
    count += count[:1]

    # Create angles of the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], [], color='grey', size=12)
    ax.set_yticklabels([])

    # Plot data
    max_val = 0
    legend_elements = []    
    
    for index, user in enumerate(df.User.unique()):
        values = list(df[df.User == user].Day_of_Week.value_counts().sort_index().values)
        values += values[:1]
        
        if len(values) < 8:
            continue
        
        # Set values between 0 and 1
        values = [(x - min(values)) / (max(values) - min(values)) + 1 for x in values]
        

        ax.plot(angles, values, linewidth=2, linestyle='solid', zorder=index, color=colors[index], alpha=0.8)
        ax.fill(angles, values, colors[index], alpha=0.1, zorder=0)

        if max(values) > max_val: max_val = max(values) # To align ytick labels
            
        legend_elements.append(Line2D([0], [0], color=colors[index], lw=4, label=user))

    # Draw ytick labels to make sure they fit properly
    for i in range(len(categories)):
        angle_rad = i/float(len(categories))*2*np.pi
        angle_deg = i/float(len(categories))*360
        ha = "right"
        if angle_rad < np.pi/2 or angle_rad > 3*np.pi/2: ha = "left"
        plt.text(angle_rad, max_val*1.15, categories[i], size=14,
                 horizontalalignment=ha, verticalalignment="center")
    
    # Legend and title
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
    plt.title('Active days of each user', y=1.2)
    
    # Save or show figure    
    if savefig:
        plt.savefig(f'results/spider_plot.png', dpi = dpi)
    else:
        plt.show()
         
def plot_messages(df, colors=None, trendline=False, savefig=False, dpi=100):
    """ Plot the weekly count of messages per user
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe containing all messages
    colors : list, default None
        List of colors to be used for the plot. 
        Matplotlib colors are chosen if None. 
    trendline : boolean, default False
        Whether are not there will be a trendline for the 
        combined count of messages
    savefig : boolean, default False
        Whether or not to save the figure instead of showing
    dpi : int, default 100
        Resolution of the figure you want to save
    
    """
        
    # Prepare data
    if not colors:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10

    df = df.set_index('Date')   
    users = {user: df[df.User == user] for user in df.User.unique()}
    
    # Resample to a week by summing
    for user in users:
        users[user] = users[user].resample('7D').count().reset_index()
    
    # Create figure and plot lines
    fig, ax = plt.subplots()
    legend_elements = []
    
    for i, user in enumerate(users):
        ax.plot(users[user].Date, users[user].Message_Raw, linewidth=3, color=colors[i])
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=4, label=user))

    # calc the trendline
    if trendline:
        x = [x for x in users[user].Date.index]
        y = users[user].Message_Raw.values
        z = np.polyfit(x, y, 5)
        p = np.poly1d(z)
        ax.plot(users[user].Date, p(x), linewidth=2, color = 'g')

    # Remove axis
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    font = {'fontname':'Comic Sans MS', 'fontsize':14}
    ax.set_ylabel('Nr of Messages', {'fontname':'Comic Sans MS', 'fontsize':14})
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, 1), loc=2, borderaxespad=0.)

    # Set size of graph
    fig.set_size_inches(20, 10)
    
    # Creating custom legend
    custom_lines = [Line2D([], [], color=colors[i], lw=4, 
                          markersize=6) for i in range(len(colors))]

    # Create horizontal grid
    ax.grid(True, axis='y')
    
    # Legend and title
    ax.legend(custom_lines, [user for user in users.keys()], bbox_to_anchor=(1.05, 1), loc=2,
              borderaxespad=0.)
    plt.title("Weekly number of messages per user", fontsize=20)
    
    if savefig:
        plt.savefig(f'results/moments.png', format="PNG", dpi=dpi)
    else:
        plt.show()
        
        
def get_words_love(row):
    """ Lambda function to return 1 when somebody said some form
    of 'I love you' to the other
    
    Parameters:
    -----------
    row: dataframe
        Used to pass in a row through pandas' function apply 
            
    Returns
    -------
    int : 0 or 1 
        1 if a user said some variation of "I love you"
        0 else
    
    """
   
    # Create all combinations of words between (not within!) lists
    words = [['houd', 'hou'], ['van'], ['je', 'jou']]
    set_words = list(itertools.product(*words))
    
    # Check if any of those sets are fully matched within a sentence
    for words in set_words:
        if all(word in row.Message_Only_Text.split(' ') for word in words):
            return 1
    
    # Check also for a combination of two words 
    # This can be expanded when necessary and easily removed
    words = [['love'], ['you']]
    set_words = list(itertools.product(*words))
    
    for words in set_words:
        if all(word in row.Message_Only_Text.split(' ') for word in words):
            return 1
    
    return 0 
        
def print_stats(df, love=False, save=False):
    """ Prints the following per user:
    * Number of messages
    * Number of words
    * Messages per hour
    * Average number of words per message
    * Average length (in characaters) per message
    * Highscore day per user (most active day)
    * How often user said "I love you"
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe of all messages per user
    love : boolean, default False
        To indicate whether or not a user wants
        to see how much the other has said 
        "I love you" to the other. Currently it works
        best for the Dutch language and somewhat for 
        the English language (limited to "I love you")
    """

    if save:
        file = open("results/stats.txt", "a")
    else:
        file = None

    # Print number of messages
    print_title('Number of Messages', file=file)
    for user in df.User.unique():
        nr_messages = len(df[df.User == user])
        print(str(nr_messages) + '\t' + user, file=file)
    print(file=file)

    # Print number of words
    print_title('Number of Words', file=file)
    for user in df.User.unique():
        nr_words = len([x for sublist in df[df.User==user].Message_Clean.values 
                           for x in sublist.split(' ')])
        print(str(nr_words) + '\t' + user, file=file)
    print(file=file)
    
    # Calculate messages per hour per user
    print_title('Messages per hour', file=file)
    for user in df.User.unique():
        start = df.Date[df[df.User == user].index[0]]
        end = df.Date[df[df.User == user].index[-1]]
        diff = end - start
        hours = diff.components[0] * 24 + diff.components[1]
        print(user + ':\t{}'.format(len(df[df.User==user])/hours), file=file)
    print(file=file)
    
    # Calculate average number of words en characters per set of messages
    df['avg_length_words'] = df.apply(lambda row: len(row.Message_Only_Text.split(" ")), 1)
    df['avg_length_charac'] = df.apply(lambda row: len(row.Message_Only_Text), 1)
    
    # Avg number of words per message
    print_title("Avg nr Words per Message", file=file)
    for user in df.User.unique():
        mean = (sum(df.loc[df.User == user, 'avg_length_words']) / 
                len(df.loc[df.User == user, 'avg_length_words']))
        print(user + ": " + str(round(mean, 2)), file=file)
    print(file=file)
    
    # Average length of message
    print_title('Avg length of Message', file=file)
    for user in df.User.unique():
        mean = (sum(df.loc[df.User == user, 'avg_length_charac']) / 
                len(df.loc[df.User == user, 'avg_length_charac']))
        print(user + ": " + str(round(mean, 2)), file=file)
    print(file=file)
    
    # Highscore Day
    print_title('Highscore Day per User', file=file)
    df['Date_only'] = df.apply(lambda x: str(x.Date).split(' ')[0], 1)
    for user in df.User.unique():
        temp = df[df.User == user].groupby(by='Date_only').count()
        temp.loc[temp['User'].idxmax()]

        print(user, file=file)
        print("Messages: \t{}".format(temp.loc[temp['User'].idxmax()].User), file=file)
        print("Day: \t\t{}".format(temp['User'].idxmax()), file=file)
        print(file=file)
    
    # Count for each row if somebody said "I love you"
    if love:   
        df['Love'] = df.apply(lambda row: get_words_love(row), axis=1)
        print_title('How often user said "I love you"', file=file)
        
        for user in df.User.unique():
            print('{0: <30}'.format(user + ':') + str(sum(df[df.User == user].Love)), file=file)


def print_timing(df, save=False):
    """ Print for each user their average response time
    and the number of times they initiated a message. 
    
    A response is recorded as such when it is within 1 day after
    the other user had sent a message. This is an assumption that
    is likely to be challenged since it is easily possible that a 
    message after an hour can be a new message while a message after 
    a day can also be a response. However, an assumption is necessary
    to define a response. 
    
    The number of times a user initiated a messages is defined
    as the message a user sent after a different user has sent 
    a message with a 1 day difference. 
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe of all messages
    
    """
    if save:
        file = open("results/timing.txt", "a")
    else:
        file = None

    # Needed later on to calculate total nr of messages
    raw_data = df.copy()
    
    # Calculate difference in time between messages and remove first row
    df['Response_Time'] = df.Date.diff()
    df = df.drop(df.index[0])

    # Get first response_time of each user consecutively
    # Each first response_time after a different user is the 
    # real response time, thus I take first()
    # then simply take response time in seconds
    df = df.groupby([(df.User != df.User.shift()).cumsum()]).first()
    df['Response_Time'] = df.apply(lambda row: row.Response_Time.total_seconds(), 1)
    
    # Remove all messages that were sent more than a day after the previous
    # Here I make the assumption that it is not a response, but a new message
    response = df[(df.Response_Time/60/60/24) < 0.5]

    # Then, for each user calculate the average response time
    print_title("Avg. Response Time in Minutes", file=file)

    for user in response.User.unique():
        minutes = round(np.mean(response.loc[response.User == user, 'Response_Time']) / 60, 2)
        print('{0: <30}'.format(user + ':') + str(minutes), file=file)
    print(file=file)
        
    # Remove all messages that were sent more than a day after the previous
    # Here I make the assumption that it is not a response, but a new message
    response = df[(df.Response_Time/60/60/24) > 1]

    # Then, for each user calculate the average response time
    print_title("Nr. Initiated Messages", file=file)
    
    for user in response.User.unique():
        nr_initiated = len(response.loc[response.User == user])
        nr_messages = len(raw_data[raw_data.User == user])
        percentage = str(round(nr_initiated / nr_messages * 100, 2))

        print('{0: <30}'.format(user + ':') + str(nr_initiated) + 
              "\t\t" + '{0: <6}'.format('('+percentage +  '%') + " of all messages)", file=file)


def print_title(title, file):
    """ Used to print titles in a certain format
    for the functions that print data
    
    Parameters:
    -----------
    title : string
        The title to print
    """
    print("#" * (len(title)+8), file=file)
    print("##  " + title + "  ##", file=file)
    print("#" * (len(title)+8), file=file)
    print(file=file)


def calendar_plot(data, year=None, how='count', column = 'User', savefig=False, dpi=100):
    """ Adjusted calendar plot from https://pythonhosted.org/calmap/
    
    Copyright (c) 2015 by Martijn Vermaat
    
    
    To do:
    * year set to None and find the minimum year
    * Choose column instead of using index
    * Set date as index

    
    Parameters:
    -----------
    year : boolean, default None
    how : string, default 'count'
        Which methods to group by the values. 
        Note, it is always by day due to the
        nature of a calendar plot. 
    column : string, default 'User'
        The column over which you either count or sum the values
        For example, count activity in a single day.
    savefig : boolean, default False
        Whether or not to save the figure instead of showing.
    dpi : int, default 100
        Resolution of the figure you want to save.
    
    
    """
    
    # Get minimum year if not given
    if year == None:
        year = data.Date.min().year
    
    # Prepare data
    data = data.set_index('Date').loc[:, column]
    
    # Resample data
    if how == 'sum':
        daily = data.resample('D').sum()
    elif how == 'count':
        daily = data.resample('D').count()
    
    vmin = daily.min()
    vmax = daily.max()

    # Fill in missing dates
    daily = daily.reindex(pd.date_range(start=str(year), end=str(year + 1), 
                                        freq='D')[:-1])

    # Put into dataframe
    # Fill is needed to created the initial raster
    daily = pd.DataFrame({'data': daily,
                           'fill': 1,
                           'day': daily.index.isocalendar().day,
                           'week': daily.index.isocalendar().week})

    # Correctly choose week and day
    daily.loc[(daily.index.month == 1) & (daily.week > 50), 'week'] = 0
    daily.loc[(daily.index.month == 12) & (daily.week < 10), 'week'] \
        = daily.week.max() + 1

    # Create data to be plotted
    plot_data = daily.pivot('day', 'week', 'data').values[::-1]
    plot_data = np.ma.masked_where(np.isnan(plot_data), plot_data)

    # Create data for the background (all days)
    fill_data = daily.pivot('day', 'week', 'fill').values[::-1]
    fill_data = np.ma.masked_where(np.isnan(fill_data), fill_data)

    # Set plotting values
    cmap='OrRd'
    linewidth=1
    linecolor = 'white'
    fillcolor='whitesmoke'

    # Draw heatmap for all days of the year with fill color.
    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    ax.pcolormesh(fill_data, vmin=0, vmax=1, cmap=ListedColormap([fillcolor]))
    ax.pcolormesh(plot_data, vmin=vmin, vmax=vmax, cmap=cmap, 
                  linewidth=linewidth, edgecolors=linecolor)

    # Limit heatmap to our data.
    ax.set(xlim=(0, plot_data.shape[1]), ylim=(0, plot_data.shape[0]))

    # # Square cells.
    ax.set_aspect('equal')

    # plt.axis('off')

    # Remove spines and ticks.
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.yaxis.set_tick_params(which='both', length=0)

    # Get ticks and labels for days and months
    daylabels = calendar.day_abbr[:]
    dayticks = range(len(daylabels))

    monthlabels = calendar.month_abbr[1:]
    monthticks = range(len(monthlabels))

    # Create label and ticks for x axis
    font = {'fontname':'Comic Sans MS', 'fontsize':20}
    ax.set_xlabel('')
    ax.set_xticks([3+i*4.3 for i in monthticks])
    # ax.set_xticks([daily.loc[datetime.date(year, i + 1, 15),:].week
    #                for i in monthticks])
    ax.set_xticklabels([monthlabels[i] for i in monthticks], ha='center', **font)

    # Create label and ticks for y axis
    font = {'fontname':'Comic Sans MS', 'fontsize':15}
    ax.set_ylabel('')
    ax.yaxis.set_ticks_position('right')
    ax.set_yticks([6 - i + 0.5 for i in dayticks])
    ax.set_yticklabels([daylabels[i] for i in dayticks], rotation='horizontal',
                       va='center', **font)
    
#     ax.set_xlim(0, 54)

    ax.set_ylabel(str(year), fontsize=52,color='#DCDCDC',fontweight='bold',
                  fontname='Comic Sans MS', ha='center')
    if savefig:
        fig.savefig(f'results/calendar_{year}.png', format="PNG", dpi=dpi)
    else:
        plt.show()
