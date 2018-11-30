from math import log
import pandas as pd
import re
import wordninja

# Build cost dictionaries, assuming Zipf's law and cost = -math.log(probability).

name_dict = "conf/name_1998+arabic.csv"
word_dict = "conf/word_109584.txt"

df = pd.read_csv(name_dict, header=0, names=['name', 'gender', 'val'])
name_words = df['name'].values.tolist()
name_words_value = df['val'].values.tolist()
name_words_cost = dict((str(word).lower(), log((index + 1) * log(len(name_words))) + name_words_value[index]) for index, word in enumerate(name_words))
name_maxword = max(len(str(word)) for word in name_words)

global_words = open(word_dict).read().split()
global_words_cost = dict((str(word).lower(), log((index + 1) * log(len(global_words)))) for index, word in enumerate(global_words))
global_maxword = max(len(str(word)) for word in global_words)

all_words = name_words + global_words
all_words_cost = dict((str(word).lower(), log((index + 1) * log(len(all_words)))) for index, word in enumerate(all_words))
all_maxword = max(len(str(word)) for word in all_words)


def infer_spaces(screen_name, word_type):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).

    screen_name = re.sub(r"[^a-zA-Z]", '', screen_name)

    def best_match(i, word_type):
        if word_type == 'NAME':
            candidates = enumerate(reversed(cost[max(0, i-name_maxword):i]))
            return min((c + name_words_cost.get(screen_name[i-k-1:i], 9e999), k+1) for k, c in candidates)
        elif word_type == 'GLOBAL':
            candidates = enumerate(reversed(cost[max(0, i - global_maxword):i]))
            return min((c + global_words_cost.get(screen_name[i - k - 1:i], 9e999), k + 1) for k, c in candidates)
        elif word_type == 'ALL':
            candidates = enumerate(reversed(cost[max(0, i - global_maxword):i]))
            return min((c + all_words_cost.get(screen_name[i - k - 1:i], 9e999), k + 1) for k, c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1, len(screen_name)+1):
        c,k = best_match(i, word_type)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(screen_name)
    while i > 0:
        c, k = best_match(i, word_type)
        assert c == cost[i]
        out.append(screen_name[i-k:i])
        i -= k

    return " ".join(reversed(out))


def get_words_list(user_name):
    str_name = str(user_name).lower()
    word_list_1 = infer_spaces(str_name, 'NAME').split()
    word_list_2 = infer_spaces(str_name, 'GLOBAL').split()
    word_list_3 = infer_spaces(str_name, 'ALL').split()
    word_list_4 = wordninja.split(str_name)

#    print word_list_1
#    print word_list_2
#    print word_list_3
#    print word_list_4

    return min([word_list_1, word_list_2, word_list_3, word_list_4], key=len)

#print get_words_list('123tommy')