import joblib
import lightgbm
import pandas as pd

class ScoringModel(object):
    def __init__(self):
        self.model = joblib.load('lgbm_model.pkl')
        self.data = pd.read_csv('demo_data.csv')
        self.features = [x for x in self.data.columns if x not in ['app_id', 'flag']]

    def predict(self, features):
        try:
            return self.model.predict_proba(features)[:, 1]
        except:            
            print('Failed to predict')
            return None