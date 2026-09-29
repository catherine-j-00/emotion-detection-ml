import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import pandas as pd
import  re
import contractions
import string
import emoji
from sklearn.metrics import(accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

def contradiction_manual(data):
  change_contradiction={
    "didn t":"did not",
    "didnt":"did not",
    "dont":"do not",
    "don t":"do not",
    "doesn t":"does not",
    "isn t":"is not",
    "aren t":"are not",
    "wasn t":"was not",
    "wasnt":"was not",
    "weren t":"were not",
    "won t":"will not",
    "can t":"cannot",
    "im":"i am",
    "ive":"i have"
    }
  for old,new in change_contradiction.items():
    data=data.replace(old,new)
  return data


def preprocessing(data):
    # url
    if(re.search(r'https?://\S+|www\.\S+',data)):
        data=re.sub(r'https?://\S+|www\.\S+','',data)


    if(re.search(r'<.*?>',data)):
        data=re.sub(r'<.*?>','',data)

    
    if(re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',data)):
        data=re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}','',data)

    
    if(re.search(r'@\w+',data)):
        data=re.sub(r'@\w+','',data)

    
    if(re.search(r'[A-Z]',data)):
        data=data.lower()

    
    if(re.search(r'^\s|\s$|\s{2,}|\t|\n',data)):
        data=" ".join(data.split())

   
    if( re.search(r"\b\w+'\w+\b",data)):
        data=contractions.fix(data)
    data=contradiction_manual(data)    

    
    if(re.search(r'[^\w\s]',data)):
       data=data.translate(str.maketrans("","",string.punctuation))

    e_count=emoji.emoji_count(data)
    pattern=r":-\)|:\)|:-\(|:\(|:D|:-D|;\)|;-\)|:P|:-P|:p|:-p|<3|:'\(|:\||:-\|"

    if(e_count>0 ):
        data=emoji.replace_emoji(data,"")
    if(re.search(pattern,data)):
        data=re.sub(pattern,"",data)

   
    if(re.search(r'\d',data)):
        data=re.sub(r"\d+","",data)

    return data

def deploy_preprocessing(data):
    # url
    if(re.search(r'https?://\S+|www\.\S+',data)):
        data=re.sub(r'https?://\S+|www\.\S+','',data)


    if(re.search(r'<.*?>',data)):
        data=re.sub(r'<.*?>','',data)

    
    if(re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',data)):
        data=re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}','',data)

    
    if(re.search(r'@\w+',data)):
        data=re.sub(r'@\w+','',data)

    
    if(re.search(r'[A-Z]',data)):
        data=data.lower()

    
    if(re.search(r'^\s|\s$|\s{2,}|\t|\n',data)):
        data=" ".join(data.split())

   
    if( re.search(r"\b\w+'\w+\b",data)):
        data=contractions.fix(data) 

    
    if(re.search(r'[^\w\s]',data)):
       data=data.translate(str.maketrans("","",string.punctuation))

    e_count=emoji.emoji_count(data)
    pattern=r":-\)|:\)|:-\(|:\(|:D|:-D|;\)|;-\)|:P|:-P|:p|:-p|<3|:'\(|:\||:-\|"

    if(e_count>0 ):
        data=emoji.replace_emoji(data,"")
    if(re.search(pattern,data)):
        data=re.sub(pattern,"",data)

   
    if(re.search(r'\d',data)):
        data=re.sub(r"\d+","",data)

    return data

import spacy
nlp=spacy.load("en_core_web_sm")
original_stopwords=nlp.Defaults.stop_words
stopwords=original_stopwords.copy()
stopwords.difference_update({'again','all','almost','already','although','always',"no","now",'none','not','nor','nothing','please'})
stopwords.difference_update({'because','both','can','cannot','could','each','either','enough','even','ever','except','further'})
stopwords.difference_update({'get','give','go','hence','here','first','full','how','however','if','indeed','keep','least'})
stopwords.difference_update({'less','may','meanwhile','might','more','moreover','most','much','must',"n't",'never','nevertheless'})
stopwords.difference_update({'often','only','other','otherwise','own','perhaps','quite','rather','really','same','see','seem'})
stopwords.difference_update({'seemed','seeming','seems','serious','should','since','so','sometimes','still','such','than'})
stopwords.difference_update({'therefore','though','thus','too','unless','until','very','well','what','whatever','when','where'})
stopwords.difference_update({'whether','which','while','who','whom','whose','why','will','without','would','yes'})

def remove_stopwords(data):
    words=data.split()
    cleaned_text=[word for word in words if word not in stopwords]
    return " ".join(cleaned_text)


def lemmatization(data):
    data=nlp(data)
    lemma_words=[token.lemma_ for token in data]
    return " ".join(lemma_words)

def contradiction_lemmatization(data):
    data=contractions.fix(data)
    data=contradiction_manual(data)
    data=lemmatization(data)
    return data

def preprocessor_stopword(data):
    data=preprocessing(data)
    data=remove_stopwords(data)
    data=lemmatization(data)
    return data

def custom_preprocessor(data):
    data=preprocessing(data)
    data=lemmatization(data)
    return data

def convert(pred_input):
    emotion_label=pd.read_csv("emotion_label_hugging.csv")
    emotion_list=dict(zip(emotion_label["label"],emotion_label['emotion']))
    pred_emotion=[emotion_list[i] for i in pred_input]
    return pred_emotion

def probability(loaded_model,text):
    # Predict class
    pred = loaded_model.predict(text)[0]

    # Predict probabilities
    prob = loaded_model.predict_proba(text)[0]
    # Load emotion labels
    emotion_label = pd.read_csv("emotion_label_hugging.csv")
    emotion_dict = dict(zip(emotion_label['label'], emotion_label['emotion']))

    # Model class order
    classes = loaded_model.classes_

    # Predicted emotion
    print(f"\nPredicted Emotion: {emotion_dict[pred]}\n")

    # Print probabilities
    print("Emotion Probabilities:")
    for label, p in zip(classes, prob):
        print(f"{emotion_dict[label]:<10} : {p:.4f} ({p*100:.2f}%)")


def Metric(y_val,y_pred):
    accuracy=accuracy_score(y_val,y_pred)*100
    print("Accuracy=",accuracy)
    precision=precision_score(y_val,y_pred,average="weighted")*100
    print("precision=",precision)
    recall=recall_score(y_val,y_pred,average="weighted")*100
    print("recall=",recall)
    f1=f1_score(y_val,y_pred,average="weighted")*100
    print("F1 score=",f1)
    macro_f1=f1_score(y_val,y_pred,average="macro")*100
    print("macro F1 Score=",macro_f1)
    cm=confusion_matrix(y_val,y_pred)
    print("Confusion Matrix:")
    print(cm)
    cr=classification_report(y_val,y_pred,zero_division=0)
    print("classification Report:")
    print(cr)