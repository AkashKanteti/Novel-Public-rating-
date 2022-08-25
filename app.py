from flask import Flask,render_template,request
import urllib.request
import pandas as pd
import numpy as np
import json
import ssl
import re
import keras
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
app = Flask(__name__)


def clean_text(sample):
    sw = set(stopwords.words('english'))
    ps = PorterStemmer()
    sample = sample.lower()
    sample = sample.replace("<br /><br />", "")
    sample = re.sub("[^a-zA-Z]+", " ", sample)
    sample = sample.split()
    sample = [ps.stem(s) for s in sample if s not in sw] # list comprehension
    sample = " ".join(sample)
    return sample

@app.route('/', methods=['GET', 'POST'])
def future():
    if request.method == "GET":
        flis=[]
        file=open('file.txt','r')
        flis=file.read().split("\n")
        return render_template("home.html", languages=flis)
    if request.method=="POST":
        title=request.form['title']
        review=request.form['review']
        sw = set(stopwords.words('english'))
        ps = PorterStemmer()
        cv = pickle.load(open("cv.pickle", 'rb'))
        tfidf = pickle.load(open("tfidf.pickle", 'rb'))
        review=clean_text(review)
        review=pd.Series([review])
        review=cv.transform(review)
        review=tfidf.transform(review)
        review.sort_indices()

        model=keras.models.load_model('my_model.h5')
        predicted=model.predict(review)
        predicted[predicted>=0.5]=1
        predicted=predicted.astype('int')
        now=str(predicted[0][0])

        df=pd.read_csv('database.csv')
        now_rating=float(df[df['Title']==title]["Rating"].values[0])
        total_rating=int(df[df['Title']==title]["User count"].values[0])
        new_rating=(now_rating*(total_rating)+int(now))/(total_rating+1)
        df.loc[df["Title"]==title,"Rating"]=new_rating
        df.loc[df["Title"]==title,"User count"]=total_rating+1
        df.to_csv('database.csv',index=None)
        return render_template("output.html",value=str(round(new_rating*10,2)))


if __name__ == '__main__':
    app.debug = True
    app.run()
