import colorsys
import pandas as pd
import numpy as np
import re
import csv
import codecs
import _cmu_tagger
import string
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from _screenname_parser import get_words_list
from PIL import Image

name_dict = "conf/name_1998+arabic.csv"
df = pd.read_csv(name_dict, header=0, names=['name', 'gender', 'val'])


# ----------------------------------------------------------------------------------------------
# Calculate Name Score
# ----------------------------------------------------------------------------------------------
def name_score(name_column):
    name_score_list = []
    for name in name_column:
        score = [0, 0]
        words_list = str(name).lower().split()

        for word in words_list:
            row_m = df[(df['name'] == word) & (df['gender'] == 'M')].index.tolist()
            row_f = df[(df['name'] == word) & (df['gender'] == 'F')].index.tolist()

            if len(row_m) + len(row_f) == 0:
                continue
            else:
                if len(row_m) != 0:
                    score[0] += df.loc[row_m[0]]['val']
                if len(row_f) != 0:
                    score[1] += df.loc[row_f[0]]['val']
                break

        if sum(score) == 0:
            name_score_list.append(0)
        else:
            name_score_list.append((score[1] - score[0]) / (max(score) * 1.0))

    return name_score_list


# ----------------------------------------------------------------------------------------------
# Calculate Screen_name Score
# ----------------------------------------------------------------------------------------------
def screen_name_score(screen_name_column):
    screen_name_score_list = []
    for name in screen_name_column:
        score = [0, 0]
        words_list = get_words_list(name)

        for word in words_list:
            row_m = df[(df['name'] == word) & (df['gender'] == 'M')].index.tolist()
            row_f = df[(df['name'] == word) & (df['gender'] == 'F')].index.tolist()

            if len(row_m) + len(row_f) == 0:
                continue
            else:
                if len(row_m) != 0:
                    score[0] += df.loc[row_m[0]]['val']
                if len(row_f) != 0:
                    score[1] += df.loc[row_f[0]]['val']
                break

        if sum(score) == 0:
            screen_name_score_list.append(0)
        else:
            screen_name_score_list.append((score[1] - score[0]) / (max(score) * 1.0))

    return screen_name_score_list


# ----------------------------------------------------------------------------------------------
# Calculate Description Score
# ----------------------------------------------------------------------------------------------

first_list = ['i', 'am', 'my', 'me', 'mine', "i'm", 'we', 'our']

# first_list = ['i', 'am', 'my', 'me', "i'm", 'guy', 'boy', 'girl', 'man', 'woman', 'lady', 'sir']
third_list = ['official']


def desc_score(desc_column):
    desc_score_list = []
    desc_words_list = []
    for desc in desc_column:
        score = 0
        description = str(desc.encode('utf-8'))
        # remove URLs
        description = str(re.sub(r"http\S+", "", description))
        # remove Hashtags
        description = str(re.sub(r"#\S+", "", description))
        # remove @ Mentions
        description = str(re.sub(r"@\S+", "", description))
        # remove special characters
        description = re.sub(r"[^a-zA-Z0-9\n\.\']", " ", description)
        # lowercase the tweet
        description = description.lower()

        if any(word in description.split() for word in third_list):
            score -= 1
        elif any(word in description.split() for word in first_list):
            score += 1

        desc_score_list.append(score)
        desc_words_list.append(len(description.split()))

    return desc_score_list, desc_words_list


# ----------------------------------------------------------------------------------------------
# Calculate Network Score
# ----------------------------------------------------------------------------------------------
def network_score(follower_column, following_column):

    network_score_list = []

    for index in range(len(follower_column)):
        follower_num = follower_column.iloc[index]
        following_num = following_column.iloc[index]

        n_score = np.log((follower_num * follower_num + 1) * 1. / (following_num + 1))
        network_score_list.append(n_score)

    return network_score_list


# ----------------------------------------------------------------------------------------------
# Calculate Profile Image Score
# ----------------------------------------------------------------------------------------------
def prof_img_score(screen_name_column, external_folder):

    prof_img_h_score_list = []
    prof_img_s_score_list = []
    prof_img_v_score_list = []

    for screen_name in screen_name_column:
	filename = screen_name + '.jpg'
        img_file = Image.open(os.path.join(external_folder, filename))
        rgb_im = img_file.convert('RGBA')

        [xs, ys] = img_file.size
        h_list = []
        s_list = []
        v_list = []

        for x in xrange(0, xs):
            for y in xrange(0, ys):
                r, g, b, a = rgb_im.getpixel((x, y))
                r /= 255.0
                g /= 255.0
                b /= 255.0
                [h, s, v] = colorsys.rgb_to_hsv(r, g, b)
                h_list.append(h)
                s_list.append(s)
                v_list.append(v)

        prof_img_h_score_list.append(sum(h_list) * 1.0 / len(h_list))
        prof_img_s_score_list.append(sum(s_list) * 1.0 / len(s_list))
        prof_img_v_score_list.append(sum(v_list) * 1.0 / len(v_list))

    return prof_img_h_score_list, prof_img_s_score_list, prof_img_v_score_list


# ----------------------------------------------------------------------------------------------
# Calculate first, interjection, emotion scores in tweets
# ----------------------------------------------------------------------------------------------
def first_inter_emo_score(screen_name_column, external_folder, params):

    df_inter = pd.read_csv('conf/interjection.csv', header=None, names=['word'])
    inter_list = df_inter['word'].tolist()

    emo_list = []
    with open('conf/emotion.csv') as csvfile:
        readcsv = csv.reader(csvfile, delimiter=',')
        next(readcsv, None)  # skip the headers
        for row in readcsv:
            emo_list.extend([str(x).lower() for x in row if len(x) > 0])

    first_score_list = []
    inter_score_list = []
    emo_score_list = []

    for screen_name in screen_name_column:
        first_tweet_num = 0
        inter_tweet_num = 0
        emo_tweet_num = 0

        filename = os.path.join(external_folder, screen_name + '.csv')

        # Read csv file into DataFrame
        df = pd.read_csv(filename, header=None, names=['id', 'date', 'tweet'], dtype=str)
        tweet_list = df['tweet'].astype(str).tolist()

        if params == 'All':
            tweet_list = tweet_list
            num = len(df)
        else:
            tweet_list = tweet_list[0:params]
            num = params

        for tweet in tweet_list:
            if any(word in tweet.lower().split() for word in first_list):
                first_tweet_num += 1
            if any(word.lower() in tweet.lower().split() for word in inter_list):
                inter_tweet_num += 1
            if any(word.lower() in tweet.lower().split() for word in emo_list):
                emo_tweet_num += 1

            first_score = first_tweet_num * 1.0 / num
            inter_score = inter_tweet_num * 1.0 / num
            emo_score = emo_tweet_num * 1.0 / num

        first_score_list.append(first_score)
        inter_score_list.append(inter_score)
        emo_score_list.append(emo_score)

    return first_score_list, inter_score_list, emo_score_list


# ----------------------------------------------------------------------------------------------
# Process tweets for cleaning
# ----------------------------------------------------------------------------------------------
def process_raw_tweets(tweet_raw_lines):

    tweet_text_lines = tweet_raw_lines

    # Remove Retweet Tags
    tweet_text_lines = [' '.join(re.sub("(RT @\S+)", " ", tweet_text_line).split()) for tweet_text_line in tweet_text_lines]

    # Remove Mentions
    tweet_text_lines = [' '.join(re.sub("(@\S+)", " ", tweet_text_line).split()) for tweet_text_line in tweet_text_lines]

    # Remove URLs
    tweet_text_lines = [' '.join(re.sub("(https?:\/\/t\.co\S*)", " ", tweet_text_line).decode("utf-8").split()) for tweet_text_line in tweet_text_lines]

    # Extract Taggers
    tweet_tagger_lines = _cmu_tagger.runtagger_parse(tweet_text_lines, run_tagger_cmd="java -XX:ParallelGCThreads=2 -Xmx500m -jar lib/ark-tweet-nlp-0.3.2.jar")

    # Filter out Taggers + Remove Stopwords + Remove Keywords + Format words + Lemmatization + Lowercase
    tweet_processed = []

    stop_words = set(stopwords.words('english'))

    wordnet_lemmatizer = WordNetLemmatizer()

    for tweet_tagger_line in tweet_tagger_lines:

        tweet_tagger_processed = []

        for tweet_tagger in tweet_tagger_line:
            if tweet_tagger[1] in ['N', 'V', 'A', 'E', 'R', '#']:
                tagger = str(tweet_tagger[0]).lower().decode('utf-8').strip(string.punctuation)
                if tagger not in stop_words:
                    if tweet_tagger[1] == 'V':
                        tagger_lem = wordnet_lemmatizer.lemmatize(tagger, 'v')
                    else:
                        tagger_lem = wordnet_lemmatizer.lemmatize(tagger)
                    if len(tagger_lem) > 3:
                        tweet_tagger_processed.append(tagger_lem)

        tweet_tagger_processed = ' '.join(tweet_tagger_processed)

        tweet_processed.append(tweet_tagger_processed)

    # Remove Duplicates
    tweet_processed = list(set(tweet_processed))

    return tweet_processed

# ----------------------------------------------------------------------------------------------
# Calculate k-top words Score
# ----------------------------------------------------------------------------------------------
def get_ktop_words(male_column, female_column, brand_column, external_folder, num):

    def word_count(column):
        word_dict = {}
        for screen_name in column:
            filename = os.path.join(external_folder, screen_name + '.csv')
            with codecs.open(filename, 'r') as rf:
                tweets = rf.readlines()
                seen = set()
                for tweet in tweets:
                    if tweet in seen: continue
                    seen.add(tweet)
                    for word in tweet.split():
                        if word not in word_dict:
                            word_dict[word] = 1
                        else:
                            word_dict[word] += 1
                    if len(seen) == 500:
                        break
        return word_dict

    def word_cut(word_dict, k):
        word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:k]
        return word_dict

    def word_count_update(word_dict):
        word_dict_cmp = {}
        for key in word_dict.keys():
            count = 2 * word_dict[key]
            if key in word_m:
                count -= word_m[key]
            if key in word_f:
                count -= word_f[key]
            if key in word_b:
                count -= word_b[key]
            word_dict_cmp[key] = count
        return word_dict_cmp

    word_m = word_count(male_column)
    word_f = word_count(female_column)
    word_b = word_count(brand_column)

    word_m_cmp = word_cut(word_count_update(word_m), num)
    word_f_cmp = word_cut(word_count_update(word_f), num)
    word_b_cmp = word_cut(word_count_update(word_b), num)

    ktop_words_dict = {}
    ktop_words_dict.update(word_m_cmp)
    ktop_words_dict.update(word_f_cmp)
    ktop_words_dict.update(word_b_cmp)
    
    print word_m_cmp
    print word_f_cmp
    print word_b_cmp

    return ktop_words_dict


# ----------------------------------------------------------------------------------------------
# Calculate k-top words Score
# ----------------------------------------------------------------------------------------------
def get_ktop_words_2class(male_column, female_column, external_folder, num):
    def word_count(column):
        word_dict = {}
        for screen_name in column:
            filename = os.path.join(external_folder, screen_name + '.csv')
            with codecs.open(filename, 'r') as rf:
                tweets = rf.readlines()
                seen = set()
                for tweet in tweets:
                    if tweet in seen: continue
                    seen.add(tweet)
                    for word in tweet.split():
                        if word not in word_dict:
                            word_dict[word] = 1
                        else:
                            word_dict[word] += 1
                    if len(seen) == 500:
                        break
        return word_dict

    def word_cut(word_dict, k):
        word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:k]
        return word_dict

    def word_count_update(word_dict):
        word_dict_cmp = {}
        for key in word_dict.keys():
            count = 2 * word_dict[key]
            if key in word_m:
                count -= word_m[key]
            if key in word_f:
                count -= word_f[key]
            word_dict_cmp[key] = count
        return word_dict_cmp

    word_m = word_count(male_column)
    word_f = word_count(female_column)

    word_m_cmp = word_cut(word_count_update(word_m), num)
    word_f_cmp = word_cut(word_count_update(word_f), num)

    ktop_words_dict = {}
    ktop_words_dict.update(word_m_cmp)
    ktop_words_dict.update(word_f_cmp)

    print word_m_cmp
    print word_f_cmp

    return ktop_words_dict


def ktop_words_score(screen_name_column, ktop_words_dict, external_folder, num):

    ktop_words_score_list = []

    for i, screen_name in enumerate(screen_name_column):
        word_total = 0
        corpus = []

        # if i % 1000 == 0:
        #    print str(i) + " users have been processed"

        filename = os.path.join(external_folder, screen_name + '.csv')
        with open(filename) as csvfile:
            readcsv = csv.reader(csvfile, delimiter=' ')
            for tweet in readcsv:
                word_total += len(tweet)
                corpus.extend(tweet)

        if word_total == 0:
            score_list = [0.0] * (num * 3)
        else:
            score_list = [corpus.count(key) for key in ktop_words_dict]
            score_list = [float(i) / word_total for i in score_list]

        ktop_words_score_list.append(score_list)

    return ktop_words_score_list
