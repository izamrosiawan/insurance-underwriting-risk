import os
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'underwriting_pipeline.joblib')

class UnderwritingEngine:
    def __init__(self):
        saved = joblib.load(MODEL_PATH)
        self.model = saved['model']
        self.coef = saved['coef']
        self.medians = saved['medians']
        self.feature_names = saved['feature_names']

    def _discretize(self, continuous_preds: np.ndarray) -> np.ndarray:
        res = np.zeros(len(continuous_preds), dtype=int)
        for i, pred in enumerate(continuous_preds):
            if pred < self.coef[0]: res[i] = 1
            elif pred < self.coef[1]: res[i] = 2
            elif pred < self.coef[2]: res[i] = 3
            elif pred < self.coef[3]: res[i] = 4
            elif pred < self.coef[4]: res[i] = 5
            elif pred < self.coef[5]: res[i] = 6
            elif pred < self.coef[6]: res[i] = 7
            else: res[i] = 8
        return res

    def assess_risk(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        if 'Product_Info_2' in df.columns:
            df['Product_Info_2_Char'] = df['Product_Info_2'].str[0].astype('category').cat.codes
            df['Product_Info_2_Num'] = df['Product_Info_2'].str[1].astype(int)
            df = df.drop(columns=['Product_Info_2'])
            
        if 'BMI_Age_Interaction' not in df.columns and 'BMI' in df.columns and 'Ins_Age' in df.columns:
            df['BMI_Age_Interaction'] = df['BMI'] * df['Ins_Age']
            
        med_cols = [c for c in df.columns if c.startswith('Medical_Keyword_')]
        if 'Medical_Keyword_Sum' not in df.columns and len(med_cols) > 0:
            df['Medical_Keyword_Sum'] = df[med_cols].sum(axis=1)
            
        for f in self.feature_names:
            if f not in df.columns:
                df[f] = self.medians[f]
                
        df = df[self.feature_names].fillna(self.medians)
        continuous_pred = self.model.predict(df)
        return self._discretize(continuous_pred)
