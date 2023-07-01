# libs for text processing
from textblob import TextBlob

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
#nltk.download('averaged_perceptron_tagger')
nltk_lemmatizer = WordNetLemmatizer()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Stop words libraries
from nltk.corpus import stopwords
nltk_stop_words = set(stopwords.words('english'))

import spacy
sp = spacy.load('en_core_web_sm')
spacy_stopwords = sp.Defaults.stop_words

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from wordcloud import STOPWORDS as wordcloud_stopwords
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # for plotting images & adjusting colors
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# libs for data processing
import pandas as pd
import numpy as np
import json

from datetime import datetime

# Don't print warnings
#import warnings
#warnings.filterwarnings('ignore')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# for getting random img
import os
from random import randint
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Journal_cloud:
  def __init__(self):
    self.df = self.read_excel_file()

  # read excel file and return dataframe
  def read_excel_file(self):
    df = pd.read_excel('journal_xlsx.xlsx')

    # set index to date time 
    df.index = df['date_time']

    # create new columns
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['time'] = df['date_time'].dt.time

    # move columns' positions
    year = df.pop('year');      df.insert(1, 'year', year, False)
    month = df.pop('month');    df.insert(2, 'month', month, False)
    day = df.pop('day');        df.insert(3, 'day', day, False)
    time = df.pop('time');      df.insert(4, 'time', time, False)
    
    return df

  # combine stopwords
  def all_stop_words(self):
    # read json file and get list of custom stopwords
    def custom_stop_words():
      with open('extra_stop_words.json', 'r', encoding="utf-8") as file_handler:
        data = json.load(file_handler)
        #print(json.dumps(data, indent=4, ensure_ascii=False))
        return data
    
    custom_stop_words_set = set(custom_stop_words())
    combine_all = nltk_stop_words | spacy_stopwords | sklearn_stopwords | wordcloud_stopwords | custom_stop_words_set
    return combine_all

  # Lemmatiser 1: simple lemmatise function
  # INPUT: text input |  OUTPUT: lemmatised string
  def lemmatiser_simple(self, txt_input):
    # Tokenizing using Textblob
    word_tokens = TextBlob(txt_input).words
    
    # get stopwords
    stopwords = self.all_stop_words()
    
    # Convert word to lowercase. Add to filtered list if the word is NOT a stopword (check your own custom dictionary as well) 
    filtered_tokens = [word.lower() for word in word_tokens if (word.lower() not in stopwords)]
    
    # lemmatize and append to new list if all character in word is alphabetical. Then join all words into a string
    lemmatized_string_txt = ' '.join([nltk_lemmatizer.lemmatize(word) for word in filtered_tokens if word.isalpha()])
    
    return lemmatized_string_txt

  # Lemmatiser 2: Lemmatise with POS (parts of speech) tag
  # INPUT: text input |  OUTPUT: lemmatised string
  def lemmatiser_POS(self, txt_input):
    # Define function to lemmatize each word with its POS tag
    def pos_tagger(nltk_tag):
      if nltk_tag.startswith('J'):
        return wordnet.ADJ
      elif nltk_tag.startswith('V'):
        return wordnet.VERB
      elif nltk_tag.startswith('N'):
        return wordnet.NOUN
      elif nltk_tag.startswith('R'):
        return wordnet.ADV
      else:
        return None

    # get stopwords
    stopwords = self.all_stop_words()

    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(txt_input)) 
    
    # we use our 'pos_tagger' function to make things simpler to understand.
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    
    # init list to hold words lemmatised with tag
    lemmatized_tokens = []

    for word, tag in wordnet_tagged:
      if tag is None:
        # if there is no available tag, convert the word to lowercase and hold it.
        lem_word = nltk_lemmatizer.lemmatize(word).lower()
      else:       
        # else use the tag to lemmatize the token. Then convert to lowercase
        lem_word = nltk_lemmatizer.lemmatize(word, tag).lower()

      # Check if word is all alphabetical & not in stopwords library. If yes, append it to lemmatised tokens
      if lem_word.isalpha() and (lem_word not in stopwords):
        lemmatized_tokens.append(lem_word)

    # join sentence into a string & return
    return " ".join(lemmatized_tokens)

  def preprocess_text_get_dict_word_freq(self, txt_input, lemmatiser='simple', k_num_of_words=5000):
    # Preprocessing function. 
    # 1) if not a stop word
    # 2) if more than 1
    # 3) convert to lower case
    # 4) lemmatize the word
    
    # get word frequency dictionary
    def get_word_freq_dict(txt_input):
      freq_dict = {}
      for item in txt_input.split():
        if (item in freq_dict):
          freq_dict[item] += 1
        else:
          freq_dict[item] = 1
      return freq_dict
    
    if lemmatiser == 'simple':
      lemmatized_string_txt = self.lemmatiser_simple(txt_input)
    elif lemmatiser == 'pos':
      lemmatized_string_txt = self.lemmatiser_POS(txt_input)

    # create a dictionary of word frequencies
    word_freq = get_word_freq_dict(lemmatized_string_txt)
    
    # sort the dictionary
    word_freq = {k: v for k, v in sorted(word_freq.items(), reverse=True, key=lambda item: item[1])}

    # Get first K items in dictionary
    dict_with_k_limit = dict(list(word_freq.items())[0: k_num_of_words])

    return dict_with_k_limit

  def generate_word_cloud(self, word_freq_dict, img_path=None, img_size=[800, 400]):
    # Generate word cloud figure function
    '''
    params:
    1) input: list of texts | 1 single line of text
    2) cloud_bg: None (default rectangle bg) | img_path
    3) size: None (default=[2048, 1080]) | [width(px), height(px)] | 'use img size'
    4) save_img: None | '(file_name)'.jpg
    ''' 
    # other nice sizes = [2048, 1080], [800, 400]
    
    
    # if there is an image path input
    if img_path is not None:
      img = Image.open(img_path)
      background_image = np.array(img)
      # create word cloud obj (use bg image, and use original img width and height)
      word_cloud_obj = WordCloud(min_word_length = 2,
                              background_color = 'black', 
                              mask = background_image, 
                              width=img_size[0], height=img_size[1])
                              #width=img.width, height=img.height)

    # if no img path input
    else:
      # create word cloud obj (use bg image, and use original img width and height)
      word_cloud_obj = WordCloud(min_word_length = 2,
                                background_color = 'black',
                                width=img_size[0], height=img_size[1])
    
    # generate the word cloud from word frequencies
    figure = word_cloud_obj.generate_from_frequencies(word_freq_dict)
    
    # draw to figure
    plt.imshow(word_cloud_obj, interpolation='bilinear')
    plt.axis(('off'))
    
    return figure 

  def get_cloud_img(self, img_name='random'):
    # Set variables
    curr_py_path = os.getcwd()
    img_folder_name = '\\word_cloud_pics'

    # get list of images
    list_img_names = os.listdir(curr_py_path + img_folder_name)

    if img_name == 'random':
      # get a random image
      img_name = list_img_names[randint(0, len(list_img_names))]

    # get file path of img and return
    return curr_py_path + img_folder_name + '\\' + img_name

  def get_combined_txt_entries_for_period(self, input_period):
    temp_df = self.df.copy()
    text = temp_df[temp_df['date_time'].dt.year == input_period]['journal'].str.cat(sep=', \n')
    return text

  def get_cloud(self, time_period, lemmatiser,k_num_of_words, image_path, img_size, save_file_name=None):
    image = self.get_cloud_img(image_path)
    
    text = self.get_combined_txt_entries_for_period(input_period = time_period)
    word_freq = self.preprocess_text_get_dict_word_freq(txt_input=text, lemmatiser=lemmatiser, k_num_of_words=k_num_of_words)
    cloud_fig = self.generate_word_cloud(word_freq_dict=word_freq, img_path=image, img_size=img_size)

    if save_file_name is not None:
      cloud_fig.to_file(f'{save_file_name}.png')
    
    return cloud_fig