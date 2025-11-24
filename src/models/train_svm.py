from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score
import numpy as np
import os
from src.features.lbp_extractor import LBPExtractor
import cv2


dir=r"data\MSU-MFSD\pics"
attack_dir=r"data\MSU-MFSD\pics\attack"
real_dir=r"data\MSU-MFSD\pics\real"
extractor=LBPExtractor()
def preprocessing(real_dir,attack_dir):
    x=[]
    y=[]
    for f in os.listdir(real_dir):
        if f.lower().endswith((".jpg",".png")):
            img=cv2.imread(os.path.join(real_dir,f))
            if img is not None:
                x.append(extractor.extract(img))
                y.append(1)
                    
    for g in os.listdir(attack_dir):
        if g.lower().endswith((".jpg",".png")):
            img=cv2.imread(os.path.join(attack_dir,g))
            if img is not None:
                x.append(extractor.extract(img))
                y.append(0)
    return np.array(x),np.array(y)
                
                    
    

def train_model(x,y):
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.2,random_state=42)
    model=svm.SVC(kernel='rbf')
    model.fit(xtr,ytr)
    
    return model,xte,yte
    
    
def predict(model,xte,yte):
    y_preds=model.predict(xte)
    
    return y_preds

if __name__=="__main__":
    x, y = preprocessing(real_dir, attack_dir)
    model, xte, yte = train_model(x, y)
    y_preds = predict(model, xte, yte)
    print("Accuracy:", accuracy_score(yte, y_preds))
    print("Precision:", precision_score(yte, y_preds))
    print("Recall:", recall_score(yte, y_preds))
    print("F1 Score:", f1_score(yte, y_preds))
    
    
    
    
    