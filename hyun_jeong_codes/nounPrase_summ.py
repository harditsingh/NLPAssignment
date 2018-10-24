import pandas as pd
import nltk
import numpy as np
import random
import csv
import collections
import matplotlib.pyplot as plt

print('Reading Data...\n')
tokens_df = pd.read_csv('pos_tag_out.csv', sep=',', encoding='utf-8')
print('Data READ\n')

def noun_phrase_extraction(text, chunk_fn = nltk.ne_chunk()):
    extracted = chunk_fn()