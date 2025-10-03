import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

class TelcoChurn:
    def __init__(self, data_path, random_state= 10):
        self.data = pd.read_csv(data_path)
        self.random_state = random_state
        self.encoders = {}

    def preprocess(self):
        df = self.data.copy()
        df = df.dropna()
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors= 'coerce').fillna(0)
        
        for col in df.select_dtypes(["object"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            df.drop(columns=["Churn"]),
            df["Churn"],
            test_size=0.3,
            stratify=df['Churn'],
            random_state=self.random_state
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
        
    def XGBmodel(self, n_estimators= 200, max_depth= 10, learning_rate = 0.005, scale_pos_weight= 1): 
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.scale_pos_weight = scale_pos_weight
        self.model = XGBClassifier(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth, 
            learning_rate = self.learning_rate,
            scale_pos_weight = self.scale_pos_weight,
            random_state = self.random_state
        )
        return self.model

    def model_fit(self):
        self.model.fit (self.X_train, self.y_train)
        joblib.dump(self.model, "model.pkl")
        joblib.dump(self.encoders, "encoders.pkl")

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        print(f"Accuracy Score: {accuracy_score(self.y_test, y_pred):.4f}\n")
        print(f"Classification Report:\n{classification_report(self.y_test, y_pred)}")