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
        self.models = self.get_trained_models()
        self.models_description = {
        "ElasticNet": "原理：ElasticNet 是一种结合了岭回归（L2正则化）和Lasso回归（L1正则化）的回归模型，通过平衡L1和L2正则化项，既能减少系数的绝对值，又能选择特征。\n结果解释：模型输出的是一个回归预测值，表示输入数据对应的目标变量的预测值。",
        "SVR": "原理：支持向量回归（SVR）通过找到一个最优的超平面，在保证误差不超过指定阈值的情况下，使预测值尽量接近真实值。SVR可以使用不同核函数来处理线性和非线性数据。\n结果解释：返回的是回归预测值，表示输入数据的目标变量的预测结果。",
        "RandomForestRegressor": "原理：随机森林回归器通过构建多个决策树进行回归预测，每棵树通过随机选择数据子集和特征子集来训练，最终结果为所有树的预测值的平均。\n结果解释：模型输出的是输入数据的回归预测值，表示目标变量的平均预测值。",
        "KNeighborsRegressor": "原理：K邻近回归器（KNN）通过计算待预测样本与训练集中样本的距离，找到最接近的K个邻居，预测结果为这些邻居的目标值的平均值。\n结果解释：模型返回的是基于K个最邻近样本的回归预测值，表示输入数据对应的目标变量预测结果。",
        "XGBRegressor": "原理：XGBoost 回归器是一种基于梯度提升的树模型，通过逐步优化的方式构建多个决策树，每棵树的构建旨在纠正前一棵树的误差，从而提高模型的整体精度。\n结果解释：返回的是回归预测值，表示输入数据对应的目标变量的预测值。"
        }
        logger.info("Regressioner initialized.")

        
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
    
    def predict(self, x: np.ndarray, target_column: str, model_name: str='XGBRegressor', norm_features: bool=True, norm_target: bool=True):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        elif (len(x.shape) == 2 and x.shape[0] != 1) or len(x.shape) > 2:
            raise ValueError("Input x should be a 1D array or a 2D array with 1 row.")
        model = self.models[target_column][model_name]
        if norm_features:
            x_sums = x.sum(axis=1, keepdims=True)
            x = x / x_sums
        y = model.predict(x)[0]
        if norm_target:
            y = self.inverse_normal_targets(target_column, y)
        y = round(min(max(y, self.minmax_record[target_column][0]), self.minmax_record[target_column][1]), 2)
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
    
    def regression(self, x: np.ndarray, model_names, norm_features: bool=True, norm_target: bool=True):
        logger.info("Regressioner started...")
        regression_results = {}
        for model_name in model_names:
            targets_results = {}
            for target_column in self.target_columns:
                y = self.predict(x, target_column, model_name, norm_features, norm_target)
                targets_results[target_column] = round(y, 2)
            regression_results[model_name] = {
                'targets_results': targets_results,
                "model_description": self.models_description[model_name]
            }
        logger.info("Regressioner finished!")
        return regression_results
            
                
def unit_test():
    data_path = '../data/ALL_data_grouped_processed.xlsx'  # Replace with your file path
    drop_columns = ['BMGs', "Chemical composition", 'cls_label']
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus(GPa)', 'Ε(%)']

    import pandas as pd
    df = pd.read_excel(data_path)
    ml_model = Regressioner(data_path, drop_columns, target_columns)
    # x = ml_model.data.get_all_features_for_target().to_numpy()
    # for target_name in target_columns:
    #     y = ml_model.predict_multi_x(x, target_name, 'XGBRegressor')
    #     min_v, max_v = ml_model.minmax_record[target_name]
    #     # 把y里面小于min_v的值设置为min_v
    #     y[y < min_v] = min_v
    #     # 把y里面大于max_v的值设置为max_v
    #     y[y > max_v] = max_v
    #     # 当df里面的target_name列是nan的时候，把y对应index的值赋到df里面，否则保留原来的值
    #     for i, v in enumerate(y):
    #         if pd.isnull(df.loc[i, target_name]):
    #             df.loc[i, target_name] = v
    # df.to_excel('../data/ALL_data_grouped_processed_predict.xlsx', index=False)
    import pandas as pd
    all_data = pd.read_excel(data_path).drop(columns=drop_columns)
    # 从里面随机选取一个数据点
    point = all_data.sample(1)
    x = point.drop(columns=target_columns).to_numpy()
    results = ml_model.regression(x, ['ElasticNet', 'SVR', 'RandomForestRegressor', 'KNeighborsRegressor', 'XGBRegressor'])
    print(results)
        
# python -m Regression.regressioner
if __name__ == '__main__':
    unit_test()