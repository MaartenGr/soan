import re
import pandas   as pd
import numpy    as np

from collections     import Counter

def import_data(file, path = ''):
    """ Import whatsapp data and transform it to a dataframe
    
    Parameters:
    -----------
    file : str
        Name of file including its extension.
    path : str, default ''
        Path to file without the file name. 
        Keep it empty if the file is in the 
        working directory.
        
    Returns:
    --------
    df : dataframe
        Dataframe of all messages
    
    """
   
    with open(path + file, encoding = 'utf-8') as outfile:
        raw_text = outfile.readlines()
        messages = {}

        # Getting all the messages for each user 
        messages_per_user = {}

        for message in raw_text: 

            # Some messages are not sent by the user, 
            # but are simply comments and therefore need to be removed
            try:
                name = message.split(' - ')[1].split(':')[0]
            except:
                continue

            # Add name to dictionary if it exists
            if name in messages:
                messages[name].append(message)
            else:
                messages[name] = [message]

    # Convert dictionary to dataframe
    df = pd.DataFrame(columns=['Message_Raw', 'User'])

    for name in messages.keys():
        df = df.append(pd.DataFrame({'Message_Raw': messages[name], 'User': name}))

    df.reset_index(inplace=True)

    return df

def clean_message(row):
    """ 
    Try to extract name, if not possible then 
    somebody didn't write a message but changed
    the avatar of the group. 
        
    """
    
    name = row.User + ': '
    
    try:
        return row.Message_Raw.split(name)[1][:-1]
    except:
        return row.Message_Raw
    
def remove_inactive_users(df, min_messages=10):
    """ Removes inactive users or users that have 
    posted very few messages. 
    
    Parameters:
    -----------
    df : pandas dataframe
        Dataframe of all messages 
    min_messages: int, default 10
        Number of minimum messages that a user must have
        
    Returns:
    --------
    df : pandas dataframe
        Dataframe of all messages
        
    """
    # Remove users that have not posted more than min_messages
    to_keep = df.groupby('User').count().reset_index()
    to_keep = to_keep.loc[to_keep['Message_Raw'] >= min_messages, 'User'].values
    df = df[df.User.isin(to_keep)]
    return df

def preprocess_data(df, min_messages=10):
    """ Preprocesses the data by executing the following steps:
    
    * Import data
    * Create column with only message, not date/name etc.
    * Create column with only text message, no smileys etc.
    * Remove inactive users
    * Remove indices of images

    Parameters:
    -----------
    df : pandas dataframe
        Raw data in pandas dataframe format  
    min_messages : int, default 10
        Number of minimum messages each user needs
        to have posted else they are removed. 
        
    Returns:
    --------
    df : pandas dataframe
        Dataframe of all messages
        
    """
    
    # Create column with only message, not date/name etc.
    df['Message_Clean'] = df.apply(lambda row: clean_message(row), axis = 1)

    # Create column with only text message, no smileys etc.
    df['Message_Only_Text'] = df.apply(lambda row: re.sub(r'[^a-zA-Z ]+', '', 
                                                          row.Message_Clean.lower()), 
                                       axis = 1)
    
    # Remove inactive users
    df = remove_inactive_users(df, min_messages)

    # Remove indices of images
    indices_to_remove = list(df.loc[df.Message_Clean.str.contains('|'.join(['<', '>'])),
                                    'Message_Clean'].index)
    df = df.drop(indices_to_remove)
    
    # Extract Time
    df['Date'] = df.apply(lambda row: row['Message_Raw'].split(' - ')[0], axis = 1)
    
    if '/' in str(df.iloc[df.index[0]].Date):
        df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    else:
        if ',' in str(df.iloc[df.index[0]].Date):
            df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
        else:
            df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    
    # Extact Day of the Week
    df['Hour'] = df.apply(lambda row: row.Date.hour, axis = 1)
    df['Day_of_Week'] = df.apply(lambda row: row.Date.dayofweek, axis = 1)
    
    # Sort values by date to keep order
    df.sort_values('Date', inplace=True)
    
    return df
    
    