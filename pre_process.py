import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize
import re
import pandas as pd


# Process 1: Reading the data
traindata_df = pd.read_csv("input/train.csv")
# print(traindata_df)

# Process 2: Clean the Data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop=set(stopwords.words('english'))

# capital to lower
traindata_df['excerpt_preprocess']=traindata_df['excerpt'].str.lower()

# remove stop words
traindata_df['excerpt_preprocess']=traindata_df['excerpt_preprocess'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

# WordNetLemmatizer
# lemma = nltk.WordNetLemmatizer()
# print(traindata_df['excerpt_preprocess'][0])
# # for index in range(len(traindata_df['excerpt_preprocess'])) :
# #     e = [lemma.lemmatize(word) for word in traindata_df['excerpt_preprocess'][index]]
#     e=" ".join(e)
#     print(e)

# record excerpt len
traindata_df['excerpt_length']=traindata_df['excerpt_preprocess'].str.len()

ave_excerpt_len = round(traindata_df['excerpt_length'].mean(),0)
# print(ave_excerpt_len)


def feature_sentence(sentence):
    max_len=0
    count=0
    total_length=0
    punct=";|!|:|;|,|-|'"
    max_punct_len=0

    sent_1 = sent_tokenize(sentence)

    # get max sentences length and max number of short sentences
    for sent in sent_1:
        punct_len=len(re.findall(punct, sent))
        if punct_len>max_punct_len:
            max_punct_len=punct_len
        if len(sent)>max_len:
            max_len=len(sent)
        total_length+=len(sent)
        count+=1
    print("Average sent Length: ",round(total_length/count,1))
    print("Max sent Length: ",max_len)
    print("Max punct Length: ",max_punct_len)


old_excerpt = traindata_df.loc[0,'excerpt']
preprocessed_excerpt= traindata_df.loc[0,'excerpt_preprocess']
feature_sentence(old_excerpt)
feature_sentence(preprocessed_excerpt)

preprocessed_train=traindata_df.to_csv('input/preprocessed_train.csv',index=0)

