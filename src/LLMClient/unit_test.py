from LLMClient.llm_ana import *
import asyncio

async def llm_eval_unit_test():
    # LLM 直接分析测试
    input_point = {
        "Composition": "Al20(CoCrCuFeMnNiTiV)80",
        "Dmax(mm)": "13",
        "Tg(K)": "423",
        "Tx(K)": "542",
        "Tl(K)": "81`2",
        "yield(MPa)": "1634",
        "Modulus(GPa)": "60",
        "Ε(%)": "4.1"
    }
    similar_bmgs = """📌 在数据集中找到了以下相似的BMGs：
    - BMGs: Al20(CoCrCuFeMnNiTiV)80 yield(MPa): 1465.0 Modulus(GPa): 190.086 Ε(%): 2.35
    - BMGs: Al11.1(CoCrCuFeMnNiTiV)88.9 yield(MPa): 1862.0 Modulus(GPa): 164.087 Ε(%): 0.95"""
    async for chunk in llm_eval(input_point, similar_bmgs):
        print(chunk, end='', flush=True)
        
async def llm_regression_unit_test():
    # LLM Regression 分析测试
    input_point = {
        "Composition": "Al20(CoCrCuFeMnNiTiV)80",
        "Dmax(mm)": "13",
        "Tg(K)": "423",
        "Tx(K)": "542",
        "Tl(K)": "81`2",
        "yield(MPa)": "1634",
        "Modulus(GPa)": "60",
        "Ε(%)": "4.1"
    }
    regression_results = {'ElasticNet': {'targets_results': {'Tg(K)': 423.74, 'Tx(K)': 477.35, 'Tl(K)': 761.89, 'Dmax(mm)': 7.58, 'yield(MPa)': 1637.47, 'Modulus(GPa)': 42.96, 'Ε(%)': 3.18}, 'model_description': '原理：ElasticNet 是一种结合了岭回归（L2正则化）和Lasso回归（L1正则化）的回归模型，通过平衡L1和L2正则化项，既能减少系数的绝对值，又能选择特征。\n结果解释：模型输出的是一个回归预测值，表示输入数据对应的目标变量的预测值。'}, 'SVR': {'targets_results': {'Tg(K)': 474.04, 'Tx(K)': 511.61, 'Tl(K)': 786.31, 'Dmax(mm)': 6.64, 'yield(MPa)': 1797.14, 'Modulus(GPa)': 50.6, 'Ε(%)': 3.83}, 'model_description': '原理：支持向量回归（SVR）通过找到一个最优的超平面，在保证误差不超过指定阈值的情况下，使预测值尽量接近真实值。SVR可以使用不同核函数来处理线性和非线性数据。\n结果解释：返回的是回归预测值，表示输入数据的目标变量的预测结果。'}, 'RandomForestRegressor': {'targets_results': {'Tg(K)': 391.0, 'Tx(K)': 452.98, 'Tl(K)': 713.58, 'Dmax(mm)': 6.48, 'yield(MPa)': 1465.96, 'Modulus(GPa)': 35.87, 'Ε(%)': 3.47}, 'model_description': '原理：随机森林回归器通过构建多个决策树进行回归预测，每棵树通过随机选择数据子集和特征子集来训练，最终结果为所有树的预测值的平均。\n结果解释：模型输出的是输入数据的回归预测值，表示目标变量的平均预测值。'}, 'KNeighborsRegressor': {'targets_results': {'Tg(K)': 395.67, 'Tx(K)': 455.67, 'Tl(K)': 721.0, 'Dmax(mm)': 6.0, 'yield(MPa)': 1735.67, 'Modulus(GPa)': 32.3, 'Ε(%)': 1.57}, 'model_description': '原理：K邻近回归器（KNN）通过计算待预测样本与训练集中样本的距离，找到最接近的K个邻居，预测结果为这些邻居的目标值的平均值。\n结果解释：模型返回的是基于K个最邻近样本的回归预测值，表示输入数据对应的目标变量预测结果。'}, 'XGBRegressor': {'targets_results': {'Tg(K)': 392.45, 'Tx(K)': 458.61, 'Tl(K)': 726.65, 'Dmax(mm)': 5.89, 'yield(MPa)': 1238.24, 'Modulus(GPa)': 50.22, 'Ε(%)': 4.3}, 'model_description': '原理：XGBoost 回归器是一种基于梯度提升的树模型，通过逐步优化的方式构建多个决策树，每棵树的构建旨在纠正前一棵树的误差，从而提高模型的整体精度。\n结果解释：返回的是回归预测值，表示输入数据对应的目标变量的预测值。'}}
    async for chunk in llm_regression(input_point, regression_results):
        print(chunk, end='', flush=True)


async def llm_anomaly_unit_test():
    # LLM Anomaly 分析测试
    input_point = {
        "Composition": "Al20(CoCrCuFeMnNiTiV)80",
        "Dmax(mm)": "13",
        "Tg(K)": "423",
        "Tx(K)": "542",
        "Tl(K)": "812",
        "yield(MPa)": "1634",
        "Modulus(GPa)": "60",
        "Ε(%)": "4.1"
    }
    anomaly_results = {'IsolationForest': {'result': 1, 'description': '原理：孤立森林通过构建一组随机决策树，将数据划分为多个区域。由于异常点与大多数数据分布差异较大，它们往往更容易被隔离，也就是它们的路径长度较短。\n结果解释：返回值为1表示数据点为正常点，-1表示数据点为异常点。'}, 'OneClassSVM': {'result': 1, 'description': '原理：一类支持向量机只使用正常数据进行训练，学习数据的边界。该模型试图找到一个高维空间中的超平面，尽可能包围正常样本。任何超出该超平面边界的数据点都被视为异常。\n结果解释：返回值为1表示数据点为正常点，-1表示数据点为异常点。'}, 'LocalOutlierFactor': {'result': -1, 'description': '原理：局部异常因子通过计算一个数据点的局部密度，并与其邻居的密度进行比较。如果某点的密度远低于其邻居的平均密度，则该点被认为是异常点。\n结果解释：返回值为1表示数据点为正常点，-1表示数据点为异常点。'}, 'GaussianMixture': {'result': -171.58963505504232, 'description': '原理：高斯混合模型假设数据是由多个高斯分布的混合组成的。通过拟合模型，可以计算每个数据点属于这些高斯分布的概率。低概率的数据点被视为异常点。\n结果解释：返回值为负对数似然，数值越大，数据点越异常。'}, 'Autoencoder': {'result': 0.046534421261337375, 'description': '原理：自编码器使用神经网络将输入数据编码为低维表示，再将其解码回原始维度。重建误差较大的样本表明其数据结构与其他数据差异较大，可能是异常点。\n结果解释：返回值为重建误差，误差越大，数据点越异常。'}}
    async for chunk in llm_anomaly(input_point, anomaly_results):
        print(chunk, end='', flush=True)

async def llm_outlier_unit_test():
    # LLM Outlier 分析测试
    input_point = {
        "Composition": "Al20(CoCrCuFeMnNiTiV)80",
        "Dmax(mm)": "13",
        "Tg(K)": "423",
        "Tx(K)": "542",
        "Tl(K)": "812",
        "yield(MPa)": "1634",
        "Modulus(GPa)": "60",
        "Ε(%)": "4.1"
    }
    outlier_results = {'BoxPlot': {'result': True, 'description': '原理：利用四分位数，定义1.5倍IQR（四分位距）之外的点为离群点。\n结果解释：如果数据点在上下四分位数1.5倍IQR之外，则被认为是离群点，返回True，否则返回False。'}, 'ZScore': {'result': False, 'description': '原理：计算数据点与均值的标准差倍数，超过设定阈值的视为离群点。\n结果解释：计算每个数据点的Z分数，若Z分数超过设定的阈值（通常为3），则视为离群点，返回True，否则返回False。'}, 'DBSCAN': {'result': True, 'description': '原理：通过密度聚类，将密度较低的点视为噪声或离群点。\n结果解释：DBSCAN聚类后，如果某个点被标记为噪声（标签为-1），则该点为离群点，返回True，否则返回False。'}, 'KMeans': {'result': False, 'description': '原理：通过聚类，计算数据点与所属簇中心的距离，距离较远的可能是离群点。\n结果解释：计算数据点与簇中心的距离，若距离超过一定阈值，则被认为是离群点，返回True，否则返回False。'}}
    # 异步生成器
    async for chunk in llm_outlier(input_point, outlier_results):
        print(chunk, end='', flush=True)
    

# python -m LLMClient.unit_test
if __name__ == "__main__":
    # asyncio.run(llm_eval_unit_test())
    # asyncio.run(llm_regression_unit_test())
    # asyncio.run(llm_anomaly_unit_test())
    asyncio.run(llm_outlier_unit_test())