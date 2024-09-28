import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

from config import logging
from Models.models import Model

logger = logging.getLogger(__name__)

class AnomalyDetector(Model):
    def __init__(self, data_path, drop_columns, target_columns):
        super().__init__(data_path, drop_columns, target_columns)
        self.scaler = StandardScaler()
        x = self.data.data.to_numpy()
        self.x = self.scaler.fit_transform(x)
        self.models = self.get_trained_models()
        self.models_description = {
            "IsolationForest": "原理：孤立森林通过构建一组随机决策树，将数据划分为多个区域。由于异常点与大多数数据分布差异较大，它们往往更容易被隔离，也就是它们的路径长度较短。\n结果解释：返回值为1表示数据点为正常点，-1表示数据点为异常点。",
            "OneClassSVM": "原理：一类支持向量机只使用正常数据进行训练，学习数据的边界。该模型试图找到一个高维空间中的超平面，尽可能包围正常样本。任何超出该超平面边界的数据点都被视为异常。\n结果解释：返回值为1表示数据点为正常点，-1表示数据点为异常点。",
            "LocalOutlierFactor": "原理：局部异常因子通过计算一个数据点的局部密度，并与其邻居的密度进行比较。如果某点的密度远低于其邻居的平均密度，则该点被认为是异常点。\n结果解释：返回值为1表示数据点为正常点，-1表示数据点为异常点。",
            "GaussianMixture": "原理：高斯混合模型假设数据是由多个高斯分布的混合组成的。通过拟合模型，可以计算每个数据点属于这些高斯分布的概率。低概率的数据点被视为异常点。\n结果解释：返回值为负对数似然，数值越大，数据点越异常。",
            "Autoencoder": "原理：自编码器使用神经网络将输入数据编码为低维表示，再将其解码回原始维度。重建误差较大的样本表明其数据结构与其他数据差异较大，可能是异常点。\n结果解释：返回值为重建误差，误差越大，数据点越异常。"
        }
        self.models_plt = {
            "IsolationForest": self.plot_isolation_forest,
            "LocalOutlierFactor": self.plot_lof,
            "GaussianMixture": self.plot_gmm
        }
        logger.info("AnomalyDetector initialized.")

    def get_trained_models(self):
        models = self._get_all_models()
        for model_name, model in models.items():
            if model_name != "Autoencoder":
                model.fit(self.x)
            else:
                y = self.x  # 自编码器的目标值是输入本身
                model.fit(self.x, y)
        return models

    def _get_all_models(self):
        models = {}
        for model_name in ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 'GaussianMixture', 'Autoencoder']:
            if model_name == 'IsolationForest':
                model = IsolationForest(contamination=0.1, n_estimators=100, max_samples=256, random_state=42)
            elif model_name == 'OneClassSVM':
                model = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
            elif model_name == 'LocalOutlierFactor':
                model = LocalOutlierFactor(n_neighbors=30, contamination=0.1)
            elif model_name == 'GaussianMixture':
                model = GaussianMixture(n_components=5, covariance_type='full', max_iter=500, random_state=42)
            elif model_name == 'Autoencoder':
                model = MLPRegressor(hidden_layer_sizes=(128, 64, 128), alpha=1e-4, max_iter=1000, random_state=42)
            models[model_name] = model
        return models

    def detect_anomalies(self, x: np.ndarray, model_name: str='IsolationForest'):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        elif len(x.shape) > 2:
            raise ValueError("Input x should be a 1D array or a 2D array.")
        
        x = self.scaler.transform(x)
        model = self.models[model_name]
        
        if model_name != 'Autoencoder':
            if model_name == 'LocalOutlierFactor':
                # LocalOutlierFactor不能直接调用predict，需要用fit_predict
                orginal_x = self.data.data.to_numpy()
                x_combined = np.vstack((orginal_x, x))
                predictions = model.fit_predict(x_combined)
                return predictions[-1]
            elif model_name == 'GaussianMixture':
                # GaussianMixture 返回的是log likelihood, 我们可以用负对数似然来表示异常度
                return -model.score_samples(x)[0]
            else:
                return model.predict(x)[0]  # 返回1是正常点，-1是异常点
        else:
            # 自编码器，计算重建误差
            y_pred = model.predict(x)
            reconstruction_error = np.mean(np.abs(y_pred - x))
            return reconstruction_error

    def plot_isolation_forest(self, x):
        model = self.models['IsolationForest']
        x = self.scaler.transform(x)
        combine_x = np.vstack((self.x, x))
        scores = model.decision_function(combine_x)
        plt.figure(figsize=(8, 6))
        plt.scatter(combine_x[:, 0], combine_x[:, 1], c=scores, cmap='coolwarm', s=20)
        # 获取最后一个点的坐标
        last_point = combine_x[-1, :]
        
        # 绘制灰色的虚线标记最后一个点，线宽为2，透明度为0.5
        plt.axhline(y=last_point[1], color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.axvline(x=last_point[0], color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.colorbar(label='Anomaly Score')
        plt.title('Isolation Forest Anomaly Detection')
        plt.tight_layout()
        return plt

    def plot_lof(self, x):
        model = self.models['LocalOutlierFactor']
        x = self.scaler.transform(x)
        combine_x = np.vstack((self.x, x))
        predictions = model.fit_predict(combine_x)
        scores = model.negative_outlier_factor_
        plt.figure(figsize=(8, 6))
        plt.scatter(combine_x[:, 0], combine_x[:, 1], c=scores, cmap='coolwarm', s=20)
        last_point = combine_x[-1, :]
        plt.axhline(y=last_point[1], color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.axvline(x=last_point[0], color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.colorbar(label='LOF Score')
        plt.title('Local Outlier Factor Anomaly Detection')
        plt.tight_layout()
        return plt

    def plot_gmm(self, x):
        model = self.models['GaussianMixture']
        x = self.scaler.transform(x)
        combine_x = np.vstack((self.x, x))
        log_likelihood = model.score_samples(combine_x)
        plt.figure(figsize=(8, 6))
        plt.scatter(combine_x[:, 0], combine_x[:, 1], c=log_likelihood, cmap='coolwarm', s=20)
        last_point = combine_x[-1, :]
        plt.axhline(y=last_point[1], color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.axvline(x=last_point[0], color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.colorbar(label='Log Likelihood')
        plt.title('Gaussian Mixture Model Anomaly Detection')
        plt.tight_layout()
        return plt
        
    def anomaly_detect(self, x, models):
        logger.info(f"Anomaly detection started...")
        anomaly_detector_results = {}
        anomaly_detector_plts = {}
        for model_name in models:
            anomaly_detector_results[model_name] = {
                'result': self.detect_anomalies(x, model_name),
                'description': self.models_description[model_name]
            }
            if model_name in self.models_plt:
                anomaly_detector_plts[model_name] = self.models_plt[model_name](x)
        logger.info(f"Anomaly detection finished!")
        return anomaly_detector_results, anomaly_detector_plts
        


def unit_test():
    data_path = '../data/ALL_data_grouped_processed_predict.xlsx'  # 替换为您的数据路径
    drop_columns = ['BMGs', "Chemical composition", 'cls_label']
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus(GPa)', 'Ε(%)']
    detector = AnomalyDetector(data_path, drop_columns, target_columns)
    import pandas as pd
    all_data = pd.read_excel(data_path).drop(columns=drop_columns)
    # 从里面随机选取一个数据点
    point = all_data.sample(1)
    x = point.to_numpy()
    models = ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 'GaussianMixture', 'Autoencoder']
    results, plts = detector.anomaly_detect(x, models)
    print(results)
    # for model_name, plt in plts.items():
    #     plt.show()


# python -m Anomaly.anomaly_detector
if __name__ == '__main__':
    unit_test()
