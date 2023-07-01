import json
import csv
import re
from datetime import datetime
import logging
# set up logger
logging.basicConfig(
  level = logging.DEBUG,
  format = '{message}',
  style = '{',
  filename = 'journal_extraction_log.log',
  filemode = 'w'
)

# return time converted to datetime obj
def get_journal_time(input_date):
  return datetime.strptime(input_date, '%Y-%m-%dT%H:%M:%S')

# Check if times are the same. If not the same, log error
def check_matching_time(journal_date_time, header_time):
  # extract time from journal datetime
  jour_time_only = journal_date_time.strftime('%I:%M %p')

  # If starting is 0, remove it
  if re.search('^0[0-9]:', jour_time_only):
    jour_time_only = re.findall('^0([0-9]:[0-9][0-9] [PA]M)', jour_time_only)[0]

  # Compare times
  if (jour_time_only == header_time.strip()):
    return
  else:
    message = f"Unmatching times for {journal_date_time.strftime('%d-%b-%Y %I:%M %p')}"
    logging.info(message)
    return

def get_days_since_last_entry(curr_entry_date, prev_entry_date):
  if (curr_entry_date is None) or (prev_entry_date is None):
    return None
  else:
    return (curr_entry_date - prev_entry_date).days

def get_journal_text(html_input):
  # get a list of sentence from html input
  list_sentences = re.findall('<p>(.*?)</p>', html_input)
  
  # init var to hold entry in one string variable. Loop over sentences and create entry txt.
  entry = ''
  for line in list_sentences:
    if line.strip() == '':
      entry += '\n\n'
    else:
      entry += line

  return entry

def get_word_count(input_text):
  num_words = 0
  word_list = input_text.split()
  for word in word_list:
    # Count +1 only if there is a letter or number
    at_least_1_alphabet = re.search('[a-zA-Z]', word)
    at_least_1_num = re.search('[0-9]', word)
    if (at_least_1_alphabet != None) or (at_least_1_num != None):
      num_words += 1
  return num_words

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_list_of_dict_entries(journal_path):
  # Init all variables to be used. 1) Lists to store all entries 2) Dictionaries 3) Time1 variable (header time)
  list_of_dict_entries_csv = []
  list_entries_w_errors = []

  # init entry dict for copying
  entry_dict = {'date_time': None, 'days_since_last_entry': None, 'word_count': None, 'tags': '', 'journal': ''}

  # var to hold date to get time days from previous entry 
  previous_entry_date = None

  with open(journal_path, 'r', encoding="utf8") as f_handle:
    data = json.load(f_handle)
    # loop over all entries + init necessary vars
    for entry in data:
      # make a try except to log date if there is an error
      try:
        # init necessary vars
        curr_entry_dict = entry_dict.copy()

        # get datetime obj and store in dict 
        curr_entry_dict['date_time'] = get_journal_time(entry['date'])
        
        # check if journal time and heading times are the same. If not matching, logging done in the function
        check_matching_time(curr_entry_dict['date_time'], entry['heading'])

        # get days since last entry
        curr_entry_dict['days_since_last_entry'] = get_days_since_last_entry(curr_entry_dict['date_time'], previous_entry_date)

        # try to get journal tags. If no tags, input 'None'
        try:
          curr_entry_dict['tags'] = entry['tags'][0]
        except IndexError:
          curr_entry_dict['tags'] = None

        # get journal entry
        curr_entry_dict['journal'] = get_journal_text(entry['html'])

        # get word count
        curr_entry_dict['word_count'] = get_word_count(curr_entry_dict['journal'])

        # change previous date (dont put this at the end cause json NOT datetime obj format will fuck it up)
        previous_entry_date = curr_entry_dict['date_time']

      # if there is an issue with entry, insert the date into list w errors
      except:
        list_entries_w_errors.append(entry['date'])

      # append this entry to data list
      list_of_dict_entries_csv.append(curr_entry_dict)

  return list_of_dict_entries_csv, list_entries_w_errors

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def add_sentiment_scores(list_csv_dicts):
  from textblob import TextBlob
  from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
  
  def get_textblob_sentiment(input_text):
    blob = TextBlob(input_text)
    sentiment = blob.polarity
    return sentiment

  def get_vader_sentiment(input_text, analyzer):
    scores = analyzer.polarity_scores(input_text)
    return scores['compound']
  
  # ~~~~~~  
  # create a SentimentIntensityAnalyzer obj
  analyzer = SentimentIntensityAnalyzer()
  
  list_csv_dicts_out = list_csv_dicts
  
  # Loop though csv dict
  for entry in list_csv_dicts_out:
    entry['TB_senti'] = get_textblob_sentiment(entry['journal'])
    entry['vader_senti'] = get_vader_sentiment(entry['journal'], analyzer)

  return list_csv_dicts_out

def organise_dict_kvp_positions(list_csv_dicts):
  list_csv_dicts_out = list_csv_dicts
  
  for entry in list_csv_dicts_out:
    entry['tags'] = entry.pop('tags')
    entry['journal'] = entry.pop('journal')

  return list_csv_dicts_out

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def save_journals_to_xlsx(list_of_dicts, file_name='journal_xlsx.xlsx'):
  import pandas as pd
  df = pd.DataFrame(data=list_of_dicts)
  df.to_excel(file_name, index=False, engine='xlsxwriter')
  logging.info(f'\nNum entries written to XLSX file: {len(list_of_dicts)}')

def save_journals_to_csv(list_of_dicts, file_name='journal_csv.csv'):
  with open(file_name, 'w', newline='', encoding='utf-8') as write_csv_file:
    field_names = ['date_time', 'days_since_last_entry', 'word_count', 'tags', 'journal']
    write_csv = csv.DictWriter(write_csv_file, fieldnames = field_names)
    write_csv.writeheader()
    for jour_dict in list_of_dicts:
      write_csv.writerow({'date_time': jour_dict['date_time'],
                          'days_since_last_entry': jour_dict['days_since_last_entry'],
                          'word_count': jour_dict['word_count'],
                          'tags': jour_dict['tags'],
                          'journal': jour_dict['journal']
                          })
  logging.info(f'\nNum entries written to CSV file: {len(list_of_dicts)}')


def conv_entries_csv_to_json(list_csv_dicts):
  return_list = list_csv_dicts.copy()

  # loop through csv list, change dict time format + append to return list
  for entry in return_list:
    entry['date_time'] = entry['date_time'].strftime('%d-%b-%Y %I:%M %p')

  return return_list

def save_journals_to_json(list_of_dicts, file_mode='w', indent_settings=2, file_name="journal_json.json"): 
  with open(file_name, file_mode, encoding="utf-8") as outfile:
    # write list opening bracket at start of file
    outfile.write('[\n')
    
    # get last item index
    last_item_index = len(list_of_dicts) -1
    
    # for each dict in list, write to file
    for i, dict in enumerate(list_of_dicts):
      json.dump(dict, outfile, indent=indent_settings, ensure_ascii=False)
      # if not last item, add ','
      if i != last_item_index:
        outfile.write(',')
        
      # write new line after every dictionary entry
      outfile.write('\n')
      
    # write list closing bracket at end of file
    outfile.write(']')
    
  logging.info(f'\nNum entries written to JSON file: {len(list_of_dicts)}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():
  # INIT ALL NECESSARY VAR NAMES HERE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # file path of json from journal app
  json_journal_file_path = 'all_entries.json'
  
  # names of output files
  out_file_name_XLSX = 'journal_xlsx' + '.xlsx'
  out_file_name_JSON = 'journal_json' + '.json'
  out_file_name_CSV = 'journal_csv' + '.csv'
  
  # Include sentiment?
  include_sentiment = True
  
  # init file formats to write. For all 3, use ['xlsx', 'json', 'csv'].
  save_formats = ['xlsx']
  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
  # run program only if formats_to_write has 'csv' or 'json'
  if ('csv' in save_formats) or ('json' in save_formats) or ('xlsx' in save_formats):
    list_dicts_csv, error_entries = get_list_of_dict_entries(json_journal_file_path)

  # If there are entries w errors, log those
  if len(error_entries) > 0 :
    for entry_date in error_entries:
      logging.info(f"ERROR IN ENTRY DATE {entry_date}. FIX BEFORE CONTINUING")

  # if no errors...
  else:
    if include_sentiment is True:
      list_dicts_csv = add_sentiment_scores(list_dicts_csv)
      list_dicts_csv = organise_dict_kvp_positions(list_dicts_csv)
    
    if 'xlsx' in save_formats:
      save_journals_to_xlsx(list_dicts_csv, file_name=out_file_name_XLSX)

    if 'csv' in save_formats:
      save_journals_to_csv(list_dicts_csv, file_name=out_file_name_CSV)

    if 'json' in save_formats:
      # convert csv file to json
      list_dicts_json = conv_entries_csv_to_json(list_dicts_csv)
      save_journals_to_json(list_dicts_json, file_name=out_file_name_JSON)
  
  logging.info('\nEnd program')

if __name__ == "__main__":
    main()