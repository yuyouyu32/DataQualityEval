import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from config import logging
from Models.models import Model


logger = logging.getLogger(__name__)

class Regressioner(Model):   
    def __init__(self, data_path, drop_columns, target_columns):
        super().__init__(data_path, drop_columns, target_columns)
        self.original_data = self.data.data.drop(columns=self.data.drop_columns)
        
    def get_trained_models(self, norm_features=True, norm_target=True):
        models = {}
        for target_column in self.target_columns:
            models[target_column] = self._get_all_models()
            x, y = self.data.get_features_for_target(target_column)
            x = x.to_numpy()
            if norm_features:
                x_sums = x.sum(axis=1, keepdims=True)
                x = x / x_sums
            y = y.to_numpy()
            if norm_target:
                y = (y - y.min()) / (y.max() - y.min())
            for model_name, model in models[target_column].items():
                model.fit(x, y)
        return models

    def _get_all_models(self):
        models = {}
        for model_name in ['ElasticNet', 'SVR', 'RandomForestRegressor', 'KNeighborsRegressor', 'XGBRegressor']:
            if model_name == 'ElasticNet':
                model = ElasticNet(alpha=0.001, l1_ratio=0.25, max_iter=1000)
            elif model_name == 'SVR':
                model = SVR(C=10, gamma=0.1)
            elif model_name == 'RandomForestRegressor':
                model = RandomForestRegressor(max_depth=None, n_estimators=50)
            elif model_name == 'KNeighborsRegressor':
                model = KNeighborsRegressor(n_neighbors=3)
            elif model_name == 'XGBRegressor':
                model = XGBRegressor(learning_rate=0.1, max_depth=5, n_estimators=500)
            models[model_name] = model
        return models
    
    def predict(self, x: np.ndarray, target_column: str, model_name: str, norm_features: bool=True, norm_target: bool=True):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        elif (len(x.shape) == 2 and x.shape[0] != 1) or len(x.shape) > 2:
            raise ValueError("Input x should be a 1D array or a 2D array with 1 row.")
        model = self.models[target_column][model_name]
        if norm_features:
            x_sums = x.sum(axis=1, keepdims=True)
            x = x / x_sums
        y = model.predict(x)
        if norm_target:
            y = self.inverse_normal_targets(target_column, y)
        y = min(max(y, self.minmax_record[target_column][0]), self.minmax_record[target_column][1])
        return y
    
    def predict_multi_x(self, x: np.ndarray, target_column: str, model_name: str, norm_features: bool=True, norm_target: bool=True):
        model = self.models[target_column][model_name]
        if norm_features:
            x_sums = x.sum(axis=1, keepdims=True)
            x = x / x_sums
        y = model.predict(x)
        if norm_target:
            y = self.inverse_normal_targets(target_column, y)
        return y
    

def unit_test():
    data_path = '../data/ALL_data_grouped_processed.xlsx'  # Replace with your file path
    drop_columns = ['BMGs', "Chemical composition", 'cls_label']
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus (GPa)', 'Ε(%)']

    import pandas as pd
    df = pd.read_excel(data_path)
    ml_model = Regressioner(data_path, drop_columns, target_columns)
    x = ml_model.data.get_all_features_for_target().to_numpy()
    for target_name in target_columns:
        y = ml_model.predict_multi_x(x, target_name, 'XGBRegressor')
        min_v, max_v = ml_model.minmax_record[target_name]
        # 把y里面小于min_v的值设置为min_v
        y[y < min_v] = min_v
        # 把y里面大于max_v的值设置为max_v
        y[y > max_v] = max_v
        # 当df里面的target_name列是nan的时候，把y对应index的值赋到df里面，否则保留原来的值
        for i, v in enumerate(y):
            if pd.isnull(df.loc[i, target_name]):
                df.loc[i, target_name] = v
    df.to_excel('../data/ALL_data_grouped_processed_predict.xlsx', index=False)
        
# python -m Regression.regressioner
if __name__ == '__main__':
    unit_test()