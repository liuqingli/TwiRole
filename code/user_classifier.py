# python Libraries

from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

# basic
import os
import re
import sys
import csv
import json
import pickle
import argparse
from twarc import Twarc
from termcolor import colored
if sys.version_info[0] == 2:
    import urllib
elif sys.version_info[0] == 3:
    from urllib.request import urlretrieve

# NLTK
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.data.path.append('../data/lib/')
nltk.download('stopwords')
nltk.download('wordnet')

# trational Classifier
import pandas as pd
import numpy as np

# deep learning
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable

# self-defined
import _cmu_tagger
import _score_calculator as sc

# Twarc Configuration
consumer_key = 'CONSUMER_KEY'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_token_secret = 'ACCESS_TOKEN_SECRET'


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False)

        self.resnet.fc = nn.Linear(512, 3)

    def forward(self, x):
        x = self.resnet(x)
        return x


def process_raw_tweets(tweet_raw_lines):
    tweet_text_lines = tweet_raw_lines

    # Remove Retweet Tags
    tweet_text_lines = [' '.join(re.sub("(RT @\S+)", " ", tweet_text_line).split()) for tweet_text_line in tweet_text_lines]

    # Remove Mentions
    tweet_text_lines = [' '.join(re.sub("(@\S+)", " ", tweet_text_line).split()) for tweet_text_line in tweet_text_lines]

    # Remove URLs
    tweet_text_lines = [' '.join(re.sub("(https?:\/\/t\.co\S*)", " ", tweet_text_line).split()) for tweet_text_line in tweet_text_lines]

    # Extract Taggers
    tweet_tagger_lines = _cmu_tagger.runtagger_parse(tweet_text_lines, run_tagger_cmd="java -XX:ParallelGCThreads=2 -Xmx500m -jar ../data/lib/ark-tweet-nlp-0.3.2.jar")

    # Filter out Taggers + Remove Stopwords + Remove Keywords + Format words + Lemmatization + Lowercase
    tweet_processed = []

    stop_words = set(stopwords.words('english'))

    wordnet_lemmatizer = WordNetLemmatizer()

    for tweet_tagger_line in tweet_tagger_lines:

        tweet_tagger_processed = []

        for tweet_tagger in tweet_tagger_line:
            if tweet_tagger[1] in ['N', 'V', 'A', 'E', 'R', '#']:
                tagger = str(tweet_tagger[0]).lower().strip(string.punctuation)
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


def user_info_crawler(screen_name, user_dir, user_profile_f, user_profileimg_f, user_tweets_f, user_clean_tweets_f):
    try:
        # crawl user profile
        # sys.stdout.write('Get user profile >> ')
        # sys.stdout.flush()

        if not os.path.exists(os.path.join(user_dir, user_profile_f)):

            t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)

            user_profile_data = t.user_lookup(ids=[screen_name], id_type="screen_name")

            for user_profile in user_profile_data:
                with open(os.path.join(user_dir, user_profile_f), 'w') as outfile:
                    json.dump(user_profile, outfile)

        # crawl user profile image
        # sys.stdout.write('Get user profile image >> ')
        # sys.stdout.flush()

        with open(os.path.join(user_dir, user_profile_f), 'r') as rf:

            user_profile_json = json.load(rf)

            if not os.path.exists(os.path.join(user_dir, user_profileimg_f)):

                # extract user profile image url
                user_profileimg_url = user_profile_json['profile_image_url']

                def image_converter(user_profileimg_url):
                    tmp_file = '../data/user/tmp' + user_profileimg_url[-4:]
                    if sys.version_info[0] == 2:
                        urllib.urlretrieve(user_profileimg_url, tmp_file)
                    elif sys.version_info[0] == 3:
                        urlretrieve(user_profileimg_url, tmp_file)
                    from PIL import Image
                    im = Image.open(tmp_file)
                    rgb_im = im.convert('RGB')
                    rgb_im.save(os.path.join(user_dir, user_profileimg_f))
                    os.remove(tmp_file)

                if user_profileimg_url:
                    user_profileimg_url = user_profileimg_url.replace('_normal', '_bigger')

                image_converter(user_profileimg_url)

        # crawl user tweets
        # sys.stdout.write('Get user tweets >> ')
        # sys.stdout.flush()

        if not os.path.exists(os.path.join(user_dir, user_tweets_f)):
            user_timeline_data = t.timeline(screen_name=screen_name)
            with open(os.path.join(user_dir, user_tweets_f), 'a') as outfile:
                for user_timeline in user_timeline_data:
                    json.dump(user_timeline, outfile)
                    outfile.write('\n')

        # clean user tweets
        # sys.stdout.write('Clean user tweets \n')
        # sys.stdout.flush()
        if not os.path.exists(os.path.join(user_dir, user_clean_tweets_f)):

            tweet_raw_lines = []
            with open(os.path.join(user_dir, user_tweets_f), 'r') as rf:
                for line in rf:
                    tweet_raw_lines.append(json.loads(line)['full_text'])

            clean_tweets = process_raw_tweets(tweet_raw_lines)

            with open(os.path.join(user_dir, user_clean_tweets_f), 'w') as wf:
                for tweet in clean_tweets:
                    if len(tweet) > 0:
                        wf.write(tweet + '\n')
            wf.close()

        return user_profile_json

    except Exception as e:
        # print(e)
        print("Could not predict user's role. Check account info, few tweets, incorrect image format...")
        # sys.exit(1)


def role_classifier(screen_name):
    try:

        user_dir = '../data/user'

        user_profile_f = screen_name + '.json'
        user_profileimg_f = screen_name + '.jpg'
        user_tweets_f = screen_name + '_tweets.json'
        user_clean_tweets_f = screen_name + '.csv'

        # If user does not exist, run crawler to get user info; Otherwise, use local data
        user_profile_json = user_info_crawler(screen_name, user_dir, user_profile_f, user_profileimg_f, user_tweets_f, user_clean_tweets_f)

        # create a one row dataframe
        user_df = pd.DataFrame(columns=['name', 'screen_name', 'desc', 'follower', 'following'])

        user_df.loc[-1] = [user_profile_json['name'], user_profile_json['screen_name'], user_profile_json['description'],
                           user_profile_json['followers_count'], user_profile_json['friends_count']]

        # ============================================
        # basic feature calculation and prediction
        # ============================================

        name_score = sc.name_score(user_df.name)
        screen_name_score = sc.screen_name_score(user_df.screen_name)
        desc_score, desc_words = sc.desc_score(user_df.desc)
        network_score = sc.network_score(user_df.follower, user_df.following)
        _, _, prof_img_v_score = sc.prof_img_score(user_df.screen_name, user_dir)
        first_score, inter_score, emo_score = sc.first_inter_emo_score(user_df.screen_name, user_dir, "All")

        TML_1_testing = pd.DataFrame()
        TML_1_testing['user'] = user_df.screen_name
        TML_1_testing['name_score'] = name_score
        TML_1_testing['screen_name_score'] = screen_name_score
        TML_1_testing['desc_score'] = desc_score
        TML_1_testing['desc_words'] = desc_words
        TML_1_testing['network_score'] = network_score
        TML_1_testing['prof_img_score'] = prof_img_v_score
        TML_1_testing['first_score'] = first_score
        TML_1_testing['inter_score'] = inter_score
        TML_1_testing['emo_score'] = emo_score

        with open('../data/model/classifier_1.pkl', 'rb') as mr:
            if sys.version_info[0] == 2:
                classifier_1 = pickle.load(mr)                                              # For Python 2
            elif sys.version_info[0] == 3:
                classifier_1 = pickle.load(mr, fix_imports=True, encoding="latin1")         # For Python 3
        classifier_1_predict = classifier_1.predict_proba(TML_1_testing[list(TML_1_testing)[1:]])

        # ============================================
        # advanced feature calculation and prediction
        # ============================================

        ktop_words_list = []
        with open('../data/conf/ktop_words.csv', 'r') as mr:
            for line in mr:
                ktop_words_list.append(line.rstrip('\n'))
        ktop_words_score = sc.ktop_words_score(user_df.screen_name, ktop_words_list, user_dir, 20)

        TML_2_testing = pd.DataFrame()
        TML_2_testing['user'] = user_df.screen_name
        for i in range(60):
            TML_2_testing['k_top_' + str(i)] = np.array(ktop_words_score)[:, i]

        with open('../data/model/classifier_2.pkl', 'rb') as mr:
            if sys.version_info[0] == 2:
                classifier_2 = pickle.load(mr)  # For Python 2
            elif sys.version_info[0] == 3:
                classifier_2 = pickle.load(mr, fix_imports=True, encoding="latin1")  # For Python 3
        classifier_2_predict = classifier_2.predict_proba(TML_2_testing[list(TML_2_testing)[1:]])

        # ============================================
        # deep feature calculation and prediction
        # ============================================

        net = ResNet18()
        net.load_state_dict(torch.load('../data/model/classifier_3.pkl', map_location='cpu'))
        net.eval()

        transform = transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        def image_loader(image_name):
            image = Image.open(image_name)
            image = transform(image).float()
            image = Variable(image)
            image = image.unsqueeze_(0)
            return image

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        classifier_3_predict = net(image_loader(os.path.join(user_dir, user_profileimg_f))).data.cpu().numpy().tolist()

        # ============================================
        # hybrid model prediction
        # ============================================

        hybrid_testing = np.concatenate((classifier_1_predict[0], classifier_2_predict[0], classifier_3_predict[0]))

        with open('../data/model/classifier_hybrid.pkl', 'rb') as mr:
            if sys.version_info[0] == 2:
                classifier_hybrid = pickle.load(mr)                                              # For Python 2
            elif sys.version_info[0] == 3:
                classifier_hybrid = pickle.load(mr, fix_imports=True, encoding="latin1")         # For Python 3
        output = classifier_hybrid.predict_proba([hybrid_testing]) * 100.0

        label_list = ['Brand', 'Female', 'Male']

        if np.argmax(output[0]) == 0:
            color = 'grey'
        elif np.argmax(output[0]) == 1:
            color = 'red'
        else:
            color = 'blue'

        # print prediction results
        print(colored('%-6s\t' % label_list[np.argmax(output[0])], color), end='')
        print(colored('[Brand: %.2f%%, Female: %.2f%%, Male: %.2f%%]' % (output[0][0], output[0][1], output[0][2]), color))

        # output_1 = classifier_1_predict[0] * 100.0
        # print('Classifier_1: \t', end='')
        # print('[Brand: %.2f%%, Female: %.2f%%, Male: %.2f%%]' % (output_1[0], output_1[1], output_1[2]))
        # output_2 = classifier_2_predict[0] * 100.0
        # print('Classifier_2: \t', end='')
        # print('[Brand: %.2f%%, Female: %.2f%%, Male: %.2f%%]' % (output_2[0], output_2[1], output_2[2]))
        # output_3 = softmax(classifier_3_predict[0]) * 100.0
        # print('Classifier_3: \t', end='')
        # print('[Brand: %.2f%%, Female: %.2f%%, Male: %.2f%%]' % (output_3[0], output_3[1], output_3[2]))

        return label_list[np.argmax(output[0])]

    except Exception as e:
        print(e)
        print("Could not predict user's role. Check account info, few tweets, incorrect image format...")
        # sys.exit(1)


def main(args):
    screen_name = args.user
    screen_name_file = args.file

    if screen_name is not None:

        sys.stdout.write("Task: %s  =>  " % screen_name)
        sys.stdout.flush()
        role_classifier(screen_name)

    else:
        with open(screen_name_file, 'r') as rf:
            screen_name_list = list(csv.reader(rf))

        for idx, screen_name in enumerate(screen_name_list):
            sys.stdout.write("Task %4d: %-15s  =>  " % (idx + 1, screen_name[0]))
            sys.stdout.flush()
            role_classifier(screen_name[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A Hybrid Model for Role-related User Classification on Twitter")
    parser.add_argument('-u', '--user', default=None, type=str, help="take a user's screen_name as input")
    parser.add_argument('-f', '--file', default=None, type=str, help="take a list of users' screen_names as input")

    args = parser.parse_args()
    if (args.user is None and args.file is None) or (args.user is not None and args.file is not None):
        parser.print_help(sys.stderr)
        print()
        sys.exit(1)
    else:
        main(args)
