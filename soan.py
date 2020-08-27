"""
Create Character Popularity Visualization
Full Example:
    python char.py --movie Frozen --extract True --fast True --prefix disney --rpath disney_reviews.json --actors False
Visualization only:
    python char.py --movie Frozen --prefix disney --rpath disney_reviews.json --npath disney_names.json --actors False
"""

import argparse
import pandas as pd
from Soan.whatsapp import helper
from Soan.whatsapp import general
from Soan.whatsapp import tf_idf
from Soan.whatsapp import emoji
from Soan.whatsapp import topic
from Soan.whatsapp import sentiment
from Soan.whatsapp import wordcloud


def parse_arguments() -> argparse.Namespace:
    """ Parse command line inputs """
    parser = argparse.ArgumentParser(description='Character')
    parser.add_argument('--file', help='The name of the movie', required=True)
    parser.add_argument('--language', help='The language of the texts', required=True)
    parser.add_argument('--hist_mask', help='Path of histogram mask', required=False, default="images/mask.png")
    parser.add_argument('--cloud_mask', help='Path of cloud mask', required=False, default="images/heart.jpg")
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    # Load data
    df = helper.import_data(f'data/{args.file}')
    df = helper.preprocess_data(df)
    user_labels = {old: new for old, new in zip(sorted(df.User.unique()), ['Her', 'Me'])}
    df.User = df.User.map(user_labels)
    users = set(df.User)

    # General plots
    general.plot_messages(df, colors=None, trendline=True, savefig=True, dpi=100)
    general.plot_day_spider(df, colors=None, savefig=True, dpi=100)
    for user in users:
        general.plot_active_days(df, savefig=user, dpi=100, user=user)
    general.plot_active_hours(df, color='#ffdfba', savefig="all", dpi=100, user='All')

    years = set(pd.DatetimeIndex(df.Date.values).year)
    for year in years:
        general.calendar_plot(df, year=year, how='count', column='index', savefig=True)
    general.print_stats(df, save=True)
    general.print_timing(df, save=True)

    # TF-IDF
    counts = tf_idf.count_words_per_user(df, sentence_column="Message_Only_Text", user_column="User")
    counts = tf_idf.remove_stopwords(counts, language=args.language, column="Word")
    unique_words = tf_idf.get_unique_words(counts, df, version='C')
    for user in users:
        tf_idf.plot_unique_words(unique_words,
                                 user=user,
                                 image_path=args.hist_mask,
                                 image_url=None,
                                 title="Me",
                                 title_color="white",
                                 title_background='#AAAAAA',
                                 width=400,
                                 height=500,
                                 save=user)

    # https://github.com/pandas-dev/pandas/issues/17892
    temp = df[['index', 'Message_Raw', 'User', 'Message_Clean', 'Message_Only_Text']].copy()
    temp = emoji.prepare_data(temp)

    # Count all emojis
    counts = emoji.count_emojis(temp, non_unicode=True)

    # Get unique emojis
    list_of_words = [word for user in counts for word in counts[user]]
    unique_emoji = emoji.get_unique_emojis(temp, counts, list_of_words)
    del temp

    # Emoji
    # emoji.print_stats(unique_emoji, counts, save=True)
    for user in users:
        emoji.plot_counts(counts, user=user, savefig=True)

    # Topic modeling
    topic.topics(df, model='lda', language=args.language, save=True)
    topic.topics(df, model='nmf', language=args.language, save=True)

    # Sentiment
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyser = SentimentIntensityAnalyzer()
    df['Sentiment'] = df.apply(lambda row: analyser.polarity_scores(row.Message_Clean)["compound"], 1)
    sentiment.plot_sentiment(df, colors=['#EAAA69', '#5361A5'], savefig=True)

    # Wordclouds
    # Counts words and create dictionary of words with counts
    counts = tf_idf.count_words_per_user(df, sentence_column="Message_Only_Text", user_column="User")
    counts = tf_idf.remove_stopwords(counts, language="dutch", column="Word")

    for user in users:
        words = counts[["Word", user]].set_index('Word').to_dict()[user]
        wordcloud.create_wordcloud(words, random_state=42, mask=args.cloud_mask,
                                   max_words=1000, max_font_size=50, scale=2,
                                   normalize_plurals=False, relative_scaling=0.5,
                                   savefig=True, name=user)


if __name__ == "__main__":
    main()
