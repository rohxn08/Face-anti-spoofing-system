from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score,precision_score,recall_score,f1_score
import numpy as np
import os
from src.features.lbp_extractor import LBPExtractor
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from src.preprocessing.preprocess import load_data_from_dirs



attack_dir=r"data\MSU-MFSD\pics\attack"
real_dir=r"data\MSU-MFSD\pics\real"
extractor=LBPExtractor()

def train_model(x,y):
    scaler=StandardScaler()
    xsca=scaler.fit_transform(x)
    xtr,xte,ytr,yte=train_test_split(xsca,y,test_size=0.2,random_state=42)
    xte=scaler.transform(xte)
    param_grid={"C":[0.1,1,10,100],
                "gamma":["scale","auto",0.01,0.001],
                "kernel":["rbf"]}
    grid=GridSearchCV(estimator=SVC(class_weight="balanced"),param_grid=param_grid,scoring="f1",cv=3,n_jobs=-1,verbose=1)
    grid.fit(xtr,ytr)
    model=grid.best_estimator_
    
    
    return model,scaler,xte,yte
    
    
def predict(model,xte,yte):
    y_preds=model.predict(xte)
    
    return y_preds

if __name__=="__main__":
    x,y=load_data_from_dirs(real_dir,attack_dir)

    model,scaler, xte, yte = train_model(x, y)
    y_preds = predict(model, xte, yte)
    print("Accuracy:", accuracy_score(yte, y_preds))
    print("Precision:", precision_score(yte, y_preds))
    print("Recall:", recall_score(yte, y_preds))
    print("F1 Score:", f1_score(yte, y_preds))