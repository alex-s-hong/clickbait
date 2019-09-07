import os
import math
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

df = pd.read_csv("./integrated.csv")

#drop the unnecessary delimiters/punctuation
s = "@\"\")\,~'\'?%/$-&_;.!|\n:(#\\[]"
for i in range(len(df)):
  tmp = df['postText'][i]
  for obj in s:
    if obj in tmp:
      tmp = tmp.replace(obj,'') #'' instead of ' '
      df.set_value(i,'postText',tmp)
      
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')
    
for i in range(len(df)):
  emofree = deEmojify(df['postText'][i])
  df.set_value(i, 'postText', emofree)
  
#remove string 'RT'
for i in range(len(df)):
  tmp = df['postText'][i]
  if 'RT' in tmp:
    tmp = tmp.replace('RT','')
    df.set_value(i,'postText',tmp)
    
#postMedia modification
x = ']['
for i in range(len(df)):
  tmp = df['postMedia'][i]
  for brac in x:
    tmp = tmp.replace(brac,'')
    df.set_value(i,'postMedia',tmp)
    
ps = PorterStemmer()

for i in range(len(df)):
  tmp = df['postText'][i]
  words = word_tokenize(tmp)
  for w in words:
    tmp = tmp.replace(w, ps.stem(w))
    df.set_value(i,'postText',tmp)
    
for i in range(len(df)):
  print(df['postText'][i])
  
media = pd.read_csv("./small_df.csv")
media_big = pd.read_csv("./big_df.csv")

med = pd.concat([media,media_big], ignore_index=True)
med.rename(columns={'PostImage': 'postMedia'}, inplace=True)

for i in range(len(df)):
  if type(df['postMedia'][i]) == type(m): #if it is str, not NaN(float type)
    tmp = df['postMedia'][i]
    tmp = tmp.replace('\'', '')
    df.set_value(i,'postMedia',tmp)
    
df = pd.merge(df, med, on='postMedia', how='inner')


#drop unnamed columns
df1 = df[df.columns.drop(list(df.filter(regex='Unnamed:')))]
list(df1)

#create binary labels for outcome
def binary_formula(x):
  if x == 'clickbait': return 1
  else: return 0

df1['truthClassInt'] = df1.apply(lambda row: binary_formula(row['truthClass']), axis = 1)


#vectorize post text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
vectorizer = CountVectorizer()  
X = vectorizer.fit_transform(df1['postText'])

temp = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())

#create new dataframe for assessing models with just text  and also for linear regression 
base = pd.merge(temp, df1['truthClassInt'], left_index=True, right_index=True)
#base1 = pd.merge(temp, df1['truthMean'], left_index=True, right_index=True)


final = pd.merge(temp, df1, left_index=True, right_index=True)
#vector form..? path twigged in ipython file
final.to_csv(r'/content/Drive/My Drive/Spring 2019/ML/sample for project/vector_form.csv', index=False)
