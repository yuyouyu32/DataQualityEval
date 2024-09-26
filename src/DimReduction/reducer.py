import numpy as np

from config import logging
from Models.models import Model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DimensionalityReducer(Model):
    def __init__(self, data_path, drop_columns, target_columns):
        super().__init__(data_path, drop_columns, target_columns)
        original_data = self.data.data.copy()
        self.original_x = original_data.drop(columns=target_columns)
        self.original_y = original_data[target_columns]
        self.models = self.reduce_original_data(self.original_x, self.original_y)
    
    def reduce_original_data(self, x, y):
        results = {}
        for method in ['PCA', 't-SNE']:
            if method == 'PCA':
                reducer_x = PCA(n_components=7, random_state=32)
                reducer = PCA(n_components=2, random_state=32)
            elif method == 't-SNE':
                reducer_x = PCA(n_components=7, random_state=32)
                reducer = TSNE(n_components=2, random_state=32)
            else:
                raise ValueError("Method must be 'PCA' or 't-SNE'")
            reduced_x = reducer_x.fit_transform(x)
            # reduced_x 和 y 横向拼接
            reduced = reducer.fit_transform(np.hstack((reduced_x, y)))
            results[method] = {
                'reducers': (reducer_x, reducer),
                'reduced_results': reduced,
                'reduced_x': reduced_x
            }
        return results
        

    def reduce_and_visualize(self, x: np.ndarray, y: np.ndarray, method: str='PCA'):
        """
        Perform dimensionality reduction and visualize the results.

        Parameters:
            method (str): Method for dimensionality reduction ('PCA' or 't-SNE').
            data (pd.DataFrame): Multi-dimensional input data.
            point (pd.Series): A single data point to highlight.

        Returns:
            plt.figure: The visualization figure.
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
        elif (len(x.shape) == 2 and x.shape[0] != 1) or len(x.shape) > 2:
            raise ValueError("Input x should be a 1D array or a 2D array with 1 row.")
        
        if method == 'PCA':
            reducer_x, reducer = self.models[method]['reducers']
        elif method == 't-SNE':
            reducer_x, reducer = self.models[method]['reducers']
        else:
            raise ValueError("Method must be 'PCA' or 't-SNE'")
        reduced_results = self.models[method]['reduced_results']

        plt.figure(figsize=(8, 8))
        plt.scatter(reduced_results[:, 0], reduced_results[:, 1], alpha=0.5, label='Data Points')
        
        reduced_x = reducer_x.transform(x)
        if method == 'PCA':
            reduced_data = reducer.transform(np.hstack((reduced_x, y)))
        elif method == 't-SNE':
            reduced_all_data = reducer.fit_transform(np.hstack((np.vstack((self.models[method]['reduced_x'], reduced_x)), np.vstack((self.original_y, y)))))
            reduced_data = reduced_all_data[-1].reshape(1, -1)
            
            
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color='red', label='Input Point', s=100, alpha=0.5)
        # 把x轴和y轴的刻度都用白色的字体显示
        plt.xticks(color='white')
        plt.yticks(color='white')
        # 显示灰色的网格，透明度为0.5，虚线，线宽为1
        plt.grid(color='gray', alpha=0.5, linestyle='--', linewidth=1)
        plt.title(f"{method} Dimensionality Reduction")
        plt.legend()
        plt.grid()
        
        return plt

def unit_test():
    data_path = '../data/ALL_data_grouped_processed_predict.xlsx'  # Replace with your file path
    drop_columns = ['BMGs', "Chemical composition", 'cls_label']
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus(GPa)', 'Ε(%)']
    rd_models = DimensionalityReducer(data_path, drop_columns, target_columns)
    import pandas as pd
    all_data = pd.read_excel(data_path).drop(columns=drop_columns)
    # 从里面随机选取一个数据点
    point = all_data.sample(1)
    x = point.drop(columns=target_columns).to_numpy()
    y = point[target_columns].to_numpy()
    rd_models.reduce_and_visualize(x, y, method='PCA')
    rd_models.reduce_and_visualize(x, y, method='t-SNE')

# python -m DimReduction.reducer
if __name__ == '__main__':
    unit_test()
    