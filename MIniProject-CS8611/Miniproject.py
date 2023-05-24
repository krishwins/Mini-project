#!/usr/bin/env python
# coding: utf-8

# #  Amazon Alexa Reviews Sentiment Analysis Machine Learning  

# ### Importing  libraries

# In[1]:


#import nltk
#nltk.download('stopwords') 
#nltk.download('punkt')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# ### Load the dataset 

# In[3]:


df = pd.read_csv('amazon_alexa.tsv',sep='\t')


# ### Reading the dataset

# In[4]:


df.head()


# In[5]:


df.tail()


# ### Descriptive Analysis

# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df["variation"].value_counts()


# ### Univariate  and Bivariate  analysis

# In[9]:


sns.countplot(x='rating', data=df)


# In[10]:


fig = plt.figure(figsize=(7,7))
colors = ("red","gold","yellowgreen","cyan","orange")
wp = {'linewidth':2, 'edgecolor':'black'}
tags = df['rating'].value_counts()
explode = (0.1,0.1,0.2,0.3,0.2)
tags.plot(kind='pie', autopct='%1.1f',colors=colors, shadow=True,
          startangle=0, wedgeprops=wp, explode=explode, label='')
plt.title('Distribution of the different ratings')
plt.show()


# In[11]:


fig = plt.figure(figsize=(30,7))
sns.countplot(x="variation",data=df)


# In[12]:


fig = plt.figure(figsize=(20,10))
sns.countplot(y="variation",data=df)


# In[13]:


df['variation'].value_counts()


# In[14]:


sns.countplot(x='feedback', data=df)
plt.show()


# In[15]:


fig = plt.figure(figsize=(7,7))
tags = df['feedback'].value_counts()
tags.plot(kind='pie', autopct='%1.1f%%', label='')
plt.title("Distribution of the different sentiments")
plt.show()


# #### Analysing the reviews of Black  Dot

# In[16]:


black_dot = df[df.variation =="Black  Dot"]


# In[17]:


black_dot.head()


# In[18]:


black_dot.shape


# In[19]:


black_dot["rating"].value_counts()


# In[20]:


sns.countplot(x="rating",data=black_dot)


# ### Cleaning and  wordcloud 

# In[21]:


for i in range(5):
    print(df['verified_reviews'].iloc[i],"\n")


# In[22]:


def data_processing(text):
    text = text.lower()
    text = re.sub(r"http\S+www\S+|https\S+", '', text, flags= re.MULTILINE)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[23]:


df.verified_reviews = df['verified_reviews'].apply(data_processing)


# In[24]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


# In[25]:


df['verified_reviews'] = df['verified_reviews'].apply(lambda x: stemming(x))


# In[26]:


for i in range(5):
    print(df['verified_reviews'].iloc[i],"\n")


# In[27]:


pos_reviews = df[df.feedback == 1]
pos_reviews.head()


# In[28]:


text = ' '.join([word for word in pos_reviews['verified_reviews']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in positive reviews', fontsize=19)
plt.show()


# In[29]:


neg_reviews = df[df.feedback==0]
neg_reviews.head()


# In[30]:


text = ' '.join([word for word in neg_reviews['verified_reviews']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in negative reviews', fontsize=19)
plt.show()


# ### Building Model

# In[31]:


X = df['verified_reviews']
Y = df['feedback']


# In[32]:


print(X)


# In[33]:


print(Y)


# In[34]:


cv = CountVectorizer()
X = cv.fit_transform(df['verified_reviews'])


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# In[36]:


print("Size of x_train: ",(x_train.shape))
print("Size of y_train: ",(y_train.shape))
print("Size of x_test: ",(x_test.shape))
print("Size of y_test: ",(y_test.shape))


# In[37]:


print(x_train)


# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[39]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


# In[40]:


print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))


# In[41]:


mnb = MultinomialNB()
mnb.fit(x_train, y_train)
mnb_pred = mnb.predict(x_test)
mnb_acc = accuracy_score(mnb_pred, y_test)
print("Test accuracy: {:.2f}%".format(mnb_acc*100))


# In[42]:


print(confusion_matrix(y_test, mnb_pred))
print("\n")
print(classification_report(y_test, mnb_pred))


# In[43]:


from sklearn.ensemble import RandomForestClassifier



# In[44]:


classifier=RandomForestClassifier(n_estimators=10,criterion="entropy")
classifier.fit(x_train,y_train)


# In[45]:


classifier_pred =classifier.predict(x_test)
classifier_acc = accuracy_score(classifier_pred, y_test)
print("Test accuracy: {:.2f}%".format(classifier_acc*100))


# In[46]:


print(confusion_matrix(y_test,classifier_pred))
print("\n")
print(classification_report(y_test,classifier_pred))


# In[47]:


from xgboost import XGBClassifier


# In[48]:


classifier =XGBClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
classifier_acc = accuracy_score(y_pred, y_test)
print("Test accuracy: {:.2f}%".format(classifier_acc*100))


# In[49]:


print(confusion_matrix(y_test,y_pred))
print("\n")
print(classification_report(y_test, y_pred))


# ### saving the model

# In[50]:


vectorizer=TfidfVectorizer()
x_t=vectorizer.fit(df['verified_reviews'])


# In[51]:


x_t


# In[52]:


x=vectorizer.transform(df['verified_reviews'])


# In[53]:


x_train, x_test, y_train, y_test = train_test_split(x,Y, test_size=0.2, random_state=42)


# In[54]:


classifier=XGBClassifier()
classifier.fit(x_train,y_train)


# In[55]:


import pickle

pickle.dump(classifier,open('model.pkl', 'wb'))


# In[56]:


import pickle

pickle.dump(vectorizer,open('tfidf.pkl', 'wb'))


# ### Deep Learning

# ### model configuration 1

# In[57]:


from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Embedding,LSTM,SpatialDropout1D


# In[58]:


tokenizer=Tokenizer(num_words=500,split=' ')
tokenizer.fit_on_texts(df['verified_reviews'])
X=tokenizer.texts_to_sequences(df['verified_reviews'])
X=pad_sequences(X)
X


# In[59]:


X.shape


# In[60]:


Y=df['feedback']


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# In[62]:


print("Size of x_train",x_train.shape)
print("Size of y_train",y_train.shape)
print("Size of x_test",x_test.shape)
print("Size of y_test",y_test.shape)


# In[63]:


model=Sequential()
model.add(Embedding(500,120,input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[64]:


history=model.fit(x_train,y_train,epochs=10,batch_size=32)


# In[65]:


scores=model.evaluate(x_test,y_test)
print("Accuracy=%0.3f"%(scores[1]*100))


# In[66]:


plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],color='r',label='loss')
plt.title("Training Loss")
plt.xlabel("number of epochs")
plt.ylabel("loss")
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],color='r',label='accuracy')
plt.title("Training Accuracy")
plt.xlabel("number of epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ### model configuration 2

# In[67]:


model=Sequential()
model.add(Embedding(500,120,input_length=X.shape[1]))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


# In[68]:


history=model.fit(x_train,y_train,epochs=10,batch_size=32)


# In[69]:


scores=model.evaluate(x_test,y_test)
print("Accuracy=%0.3f"%(scores[1]*100))


# In[70]:


plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.plot(history.history['loss'],color='r',label='loss')
plt.title("Training Loss")
plt.xlabel("number of epochs")
plt.ylabel("loss")
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'],color='r',label='accuracy')
plt.title("Training Accuracy")
plt.xlabel("number of epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

