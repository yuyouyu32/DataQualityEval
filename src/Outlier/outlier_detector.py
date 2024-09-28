import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D

from config import logging
from Models.models import Model
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class OutlierDetector(Model):
    def __init__(self, data_path, drop_columns, target_columns):
        super().__init__(data_path, drop_columns, target_columns)
        self.scaler = StandardScaler()
        x = self.data.data.to_numpy()
        self.x = self.scaler.fit_transform(x)
        self.methods = self.get_trained_methods()
        self.feature_columns = list(self.data.data.columns)
        print(self.feature_columns)
        self.methods_description = {
            "BoxPlot": "原理：利用四分位数，定义1.5倍IQR（四分位距）之外的点为离群点。\n结果解释：如果数据点在上下四分位数1.5倍IQR之外，则被认为是离群点，返回True，否则返回False。",
            "ZScore": "原理：计算数据点与均值的标准差倍数，超过设定阈值的视为离群点。\n结果解释：计算每个数据点的Z分数，若Z分数超过设定的阈值（通常为3），则视为离群点，返回True，否则返回False。",
            "DBSCAN": "原理：通过密度聚类，将密度较低的点视为噪声或离群点。\n结果解释：DBSCAN聚类后，如果某个点被标记为噪声（标签为-1），则该点为离群点，返回True，否则返回False。",
            "KMeans": "原理：通过聚类，计算数据点与所属簇中心的距离，距离较远的可能是离群点。\n结果解释：计算数据点与簇中心的距离，若距离超过一定阈值，则被认为是离群点，返回True，否则返回False。"
        }
        self.methods_plt = {
            "BoxPlot": self.plot_boxplot,
            "ZScore": self.plot_zscore,
            "DBSCAN": self.plot_dbscan,
            "KMeans": self.plot_kmeans
        }
        logger.info("OutlierDetector initialized.")

    def get_trained_methods(self):
        methods = self._get_all_methods()
        for method_name, method in methods.items():
            if method_name in ['DBSCAN', 'KMeans']:
                method.fit(self.x)
        return methods

    def _get_all_methods(self):
        methods = {}
        for method_name in ['BoxPlot', 'ZScore', 'DBSCAN', 'KMeans']:
            if method_name == 'BoxPlot':
                method = None  # No fitting required
            elif method_name == 'ZScore':
                method = None  # No fitting required
            elif method_name == 'DBSCAN':
                method = DBSCAN(eps=0.5, min_samples=5)
            elif method_name == 'KMeans':
                method = KMeans(n_clusters=3, random_state=42)
            methods[method_name] = method
        return methods

    def detect_outliers(self, x: np.ndarray, method_name: str='BoxPlot'):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        elif len(x.shape) > 2:
            raise ValueError("Input x should be a 1D array or a 2D array.")

        x = self.scaler.transform(x)
        method = self.methods[method_name]

        if method_name == 'BoxPlot':
            # Implement Box Plot method
            Q1 = np.percentile(self.x, 25, axis=0)
            Q3 = np.percentile(self.x, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (x < lower_bound) | (x > upper_bound)
            is_outlier = np.any(outlier_mask)
            return is_outlier
        elif method_name == 'ZScore':
            # Compute z-score for x
            z_scores = np.abs((x - np.mean(self.x, axis=0)) / np.std(self.x, axis=0))
            threshold = 3  # Common threshold
            outlier_mask = z_scores > threshold
            is_outlier = np.any(outlier_mask)
            return is_outlier
        elif method_name == 'DBSCAN':
            # Re-run DBSCAN on combined data
            x_combined = np.vstack((self.x, x))
            method = DBSCAN(eps=0.5, min_samples=5)
            labels = method.fit_predict(x_combined)
            is_noise = labels[-1] == -1
            return is_noise
        elif method_name == 'KMeans':
            # Compute distance to cluster centers
            cluster_labels = method.predict(x)
            cluster_center = method.cluster_centers_[cluster_labels]
            distances = np.linalg.norm(x - cluster_center, axis=1)
            # Determine threshold
            distances_all = np.linalg.norm(self.x - method.cluster_centers_[method.labels_], axis=1)
            threshold = np.mean(distances_all) + 2 * np.std(distances_all)
            is_outlier = distances > threshold
            return is_outlier[0]
        else:
            raise ValueError(f"Unknown method: {method_name}")

    def plot_boxplot(self, x):
        x = self.scaler.transform(x)
        x_combined = np.vstack((self.x, x))
        plt.figure(figsize=(8, 6))
        plt.boxplot(x_combined)
        last_point = x_combined[-1, :]
        plt.scatter(np.arange(1, x_combined.shape[1] + 1), last_point, c='red', s=50, label='Input Point', alpha=0.6)
        plt.title('Box Plot for Outlier Detection')
        plt.xlabel('Features')
        plt.legend()
        # self.feature_columns是x轴的标签, 字体设置很小，角度为45度，对齐方式为右对齐
        plt.xticks(np.arange(1, x_combined.shape[1] + 1), self.feature_columns, rotation=45, fontsize=4, ha='right')
        plt.ylabel('Value')
        plt.tight_layout()
        return plt

    def plot_zscore(self, x):
        x = self.scaler.transform(x)
        x_combined = np.vstack((self.x, x))
        z_scores = np.abs((x_combined - np.mean(self.x, axis=0)) / np.std(self.x, axis=0))
        plt.figure(figsize=(8, 6))
        for i in range(z_scores.shape[1]):
            plt.plot(z_scores[:, i], label=self.feature_columns[i])        
        
        plt.axhline(y=3, color='r', linestyle='--', label='Threshold')
        plt.legend(fontsize=4, loc='upper right')
        plt.title('Z-Score for Outlier Detection')
        plt.xlabel('Sample Index')
        plt.ylabel('Z-Score')
        plt.tight_layout()
        return plt

    def plot_dbscan(self, x):
        # 标准化输入数据
        x = self.scaler.transform(x)
        # 将输入数据与已有数据结合
        x_combined = np.vstack((self.x, x))
        # 重新运行 DBSCAN
        method = DBSCAN(eps=0.5, min_samples=5)
        labels = method.fit_predict(x_combined)
        unique_labels = set(labels)
        
        # 使用不同的颜色来区分聚类
        colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]
        
        plt.figure(figsize=(8, 6))
        
        # 绘制所有点
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # 噪声点
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xy = x_combined[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6, alpha=0.6)
        
        # 突出绘制最后一个点
        plt.plot(x_combined[-1, 0], x_combined[-1, 1], 'ro', markersize=8, label='Input Point', alpha=0.6)

        plt.title('DBSCAN Clustering for Outlier Detection')
        plt.tight_layout()
        plt.legend()
        return plt


    def plot_kmeans(self, x):
        # 标准化输入数据
        x = self.scaler.transform(x)
        # 将输入数据与已有数据结合
        x_combined = np.vstack((self.x, x))
        
        # 运行 KMeans
        method = KMeans(n_clusters=3, random_state=42)
        method.fit(self.x)
        labels = method.predict(x_combined)
        cluster_centers = method.cluster_centers_
        
        plt.figure(figsize=(8, 6))
        
        # 绘制所有点
        plt.scatter(x_combined[:-1, 0], x_combined[:-1, 1], c=labels[:-1], cmap='viridis', s=50, alpha=0.6)
        
        # 突出绘制最后一个点
        plt.scatter(x_combined[-1, 0], x_combined[-1, 1], c='red', s=80, marker='o', label='Input Point', alpha=0.6)
        
        # 绘制聚类中心
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='blue', s=100, alpha=0.75, label='Cluster Centers', marker='^')

        plt.title('KMeans Clustering for Outlier Detection')
        plt.tight_layout()
        plt.legend()
        return plt


    def outlier_detect(self, x, methods):
        logger.info(f"Outlier detection started...")
        outlier_detector_results = {}
        outlier_detector_plts = {}
        for method_name in methods:
            outlier_detector_results[method_name] = {
                'result': self.detect_outliers(x, method_name),
                'description': self.methods_description[method_name]
            }
            if method_name in self.methods_plt:
                outlier_detector_plts[method_name] = self.methods_plt[method_name](x)
        logger.info(f"Outlier detection finished!")
        return outlier_detector_results, outlier_detector_plts

def unit_test():
    data_path = '../data/ALL_data_grouped_processed_predict.xlsx'  # 替换为您的数据路径
    drop_columns = ['BMGs', "Chemical composition", 'cls_label']
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus(GPa)', 'Ε(%)']
    detector = OutlierDetector(data_path, drop_columns, target_columns)
    import pandas as pd
    all_data = pd.read_excel(data_path).drop(columns=drop_columns)
    # 从里面随机选取一个数据点
    point = all_data.sample(1)
    x = point.to_numpy()
    methods = ['BoxPlot', 'ZScore', 'DBSCAN', 'KMeans']
    results, plts = detector.outlier_detect(x, methods)
    print(results)
    for method_name, plt in plts.items():
        plt.show()

# python -m Outlier.outlier_detector
if __name__ == '__main__':
    unit_test()
