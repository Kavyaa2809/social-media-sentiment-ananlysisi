# social-media-sentiment-ananlysisi

# Social Media Sentiment Analysis: Data Processing and Visualization.
## Importing Libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from wordcloud import WordCloud
## Importing Dataset
df = pd.read_csv(r'sentimentdataset.csv')
df.head()
## Data Analysis
df.info()
df.nunique()
df.describe()
df.isnull().sum()
# Sentiment Distribution
print(df['Sentiment'].value_counts())
## Data Cleaning
df = df.drop(['Unnamed: 0','Unnamed: 0.1','Timestamp'], axis = 1)
df.head()
df['Country'] = df['Country'].str.strip()
df['Platform'] = df['Platform'].str.strip()
## EDA (Exploratory Data Analysis)
plt.figure(figsize = (30,5))
sns.countplot(data=df, x='Sentiment')
plt.title('Sentiment Distribution',color = 'blue')
plt.xticks(rotation = 90)
plt.show()
sns.countplot(data = df, x='Platform',color=None)
plt.title('Social Media Platforms')
plt.show()
fig = plt.figure(figsize = (15,8))
plat_bar = sns.countplot(df, x= 'Month', hue = 'Platform')
plt.xlabel('Months', fontsize = 12)
plt.ylabel('Count of Records', fontsize = 12)
plt.title('Number of Records for each Month', fontsize = 14)
plt.show()
fig = plt.figure(figsize = (6,5))
sns.histplot(data = df, x = 'Likes', y = 'Platform', color = 'skyblue', bins=10)
plt.title('Likes Histogram', fontsize = 16, color  = 'purple')
plt.xlabel('Number of Likes', fontsize = 12, color  = 'purple')
plt.yticks(rotation=45, )
plt.show()
fig = plt.figure(figsize = (15,8))
sns.histplot(data = df, x = 'Retweets', color = 'purple', bins=10)
plt.title('Retweets Histogram', fontsize = 16)
plt.xlabel('Number of Retweets', fontsize = 12)
plt.show()
text = " ".join(review for review in df.Text)
wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="white").generate(text)

plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Word Cloud of Text')
plt.show()
