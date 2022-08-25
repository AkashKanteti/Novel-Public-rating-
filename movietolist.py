import pandas as pd
import numpy as np
flis=[]
file=open('file.txt','r')
flis=file.read().split("\n")

df=pd.DataFrame(flis,columns=['Title'])
df['Rating']=0
df['User count']=0
df.to_csv('database.csv')
