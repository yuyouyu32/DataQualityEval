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
    response = ""
    async for chunk in llm_eval(input_point, similar_bmgs):
        print(chunk, end='', flush=True)
        response += chunk
    return response
        
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
    response = ""
    async for chunk in llm_regression(input_point, regression_results):
        print(chunk, end='', flush=True)
        response += chunk
    return response


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
    response = ""
    async for chunk in llm_anomaly(input_point, anomaly_results):
        print(chunk, end='', flush=True)
        response += chunk
    return response

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
    response = ""
    async for chunk in llm_outlier(input_point, outlier_results):
        print(chunk, end='', flush=True)
        response += chunk
    return response
    
async def llm_summary_unit_test():
    llm_ana_results = {
    "eval_llm_output": "### 数据质量评估过程\n\n#### 1. RULE评估\n\n- **平均原子尺寸差异**：成分为Al20(CoCrCuFeMnNiTiV)80。由此结构看，存在多种元素，这些元素的原子尺寸差异可能超过12%，可能有助于非晶结构形成。\n\n- **混合熵**：合金成分八元(Al, Co, Cr, Cu, Fe, Mn, Ni, Ti, V)，有多种主要元素，混合熵较高。因此评分最高。\n\n- **混合热**：需要具体计算或查阅文献来确定具体的混合热，一般来说，许多过渡金属对如上述组合倾向于负的混合热。\n\n- **模量**：杨氏模量为60 GPa，虽然不如参考数据中的高，但由于存在多样元素，这种合金潜在的变形能力可能符合某些用途。\n\n- **Tg/Tl比值**：Tg = 423 K，Tx = 542 K。需要Tg与液相温度(Tl)计算比值。然而此处缺少Tl，假设合理的Tl近似值(通常比Tx高)，可以进行初步估计，该比值需要接近或大于0.6。\n\n- **晚期过渡金属**：包含Co, Cr, Ni等过渡金属，与Ti组合，有助于非晶结构的形成。\n\n- **元素复杂性**：含有九种元素，增加了合金的复杂性，这有利于非晶态形成。\n\n#### 2. 数据比较\n\n与已知BMG数据对比：\n- 样品的模量（60 GPa）比参考的要低。\n- Yield强度（1634 MPa）在参考范围内。\n- 德裕应变（4.1%）略高于一些已知数据，可能意味着该合金具有更好的变形能力。\n\n#### 3. 相关知识与查询\n\n- 参考文献表明，Al基合金难以形成大块金属玻璃，但在高混合熵条件下，形成非晶态的可能性增加。\n- 高E%意味着潜在的塑性变形能力，但需要更多实验验证。\n\n#### 4. 打分\n\n综合评估以上因素：\n- 规则符合性：多个规则的正面指标（特别是混合熵和成分复杂性）。\n- 数据信息稍显不完整（比如缺少Tl）。\n- 参考模型表现档次合理，具备某些相似性。\n\n**评分：0.8**\n\n这个评分说明该数据具有较好的质量，符合许多BMG的理论标准和部分实验数据，不过由于某些数据缺失，还有待更多的验证。",
    "regression_llm_output": "### 综合评估报告\n\n#### 步骤 1: 收集并整理不同模型的预测结果\n\n输入数据点性能:\n- Tg(K) = 423.0\n- Tx(K) = 542.0\n- Tl(K) 未给出\n- Dmax(mm) = 13.0\n- 屈服强度(MPa) = 1634.0\n- 杨氏模量(GPa) = 60.0\n- 断裂延伸率(%) = 4.1\n\n模型预测结果:\n- **ElasticNet**: Tg = 423.74, Tx = 477.35, Tl = 761.89, Dmax = 7.58, 屈服强度 = 1637.47, 杨氏模量 = 42.96, 断裂延伸率 = 3.18\n- **SVR**: Tg = 474.04, Tx = 511.61, Tl = 786.31, Dmax = 6.64, 屈服强度 = 1797.14, 杨氏模量 = 50.6, 断裂延伸率 = 3.83\n- **RandomForestRegressor**: Tg = 391.0, Tx = 452.98, Tl = 713.58, Dmax = 6.48, 屈服强度 = 1465.96, 杨氏模量 = 35.87, 断裂延伸率 = 3.47\n- **KNeighborsRegressor**: Tg = 395.67, Tx = 455.67, Tl = 721.0, Dmax = 6.0, 屈服强度 = 1735.67, 杨氏模量 = 32.3, 断裂延伸率 = 1.57\n- **XGBRegressor**: Tg = 392.45, Tx = 458.61, Tl = 726.65, Dmax = 5.89, 屈服强度 = 1238.24, 杨氏模量 = 50.22, 断裂延伸率 = 4.3\n\n#### 步骤 2: 对比各模型预测结果的一致性\n\n1. **一致性分析**:\n   - **Tg**: 模型预测值范围在391.0到474.04，各模型差异较大。\n   - **Tx**: 模型预测值范围在452.98到511.61，各模型差异明显。\n   - **Dmax**: 模型预测值范围在5.89到7.58，显著低于输入数据的13.0。\n   - **屈服强度**: 模型预测值范围在1238.24到1797.14，与输入数据接近，但差异仍存在。\n   - **杨氏模量**: 模型预测值范围在32.3到50.6，显著低于输入的60.0。\n   - **断裂延伸率**: 模型预测值变化范围在1.57到4.3，接近输入值。\n\n#### 步骤 3: 分析模型特性对预测结果的影响\n\n- **ElasticNet**: 强调特征选择，预测结果对于输入较为平滑，有一定信赖度，但对小数据样本特点不敏感。\n- **SVR**: 对非线性关系处理较好，但可能在计算中存在过拟合，反映出不稳定的预测。\n- **RandomForestRegressor**: 表现出对数据样本变化较敏感，预测偏差可能缘于输入数据的异质性。\n- **KNeighborsRegressor**: 对邻近样本的依赖性强，容易受到训练集样本分布的影响。\n- **XGBRegressor**: 通过逐步优化，可能存在偏差调整，反映较稳定但受数据中心集的影响。\n\n#### 步骤 4: 根据统计学原理评估输入的BMG数据点性能的可靠性和可信性\n\n1. **可靠性评估**:\n   - 大多数模型在Tg、Tx和Dmax上的预测明显低于输入值，表明输入可能具有实验误差或数值异常。\n   - 屈服强度的预测相对稳定，但不一致性仍存在。\n   - 杨氏模量和Dmax出现显著差异，可能因为模型未有效捕捉特征之间的关系。\n\n2. **可信性分析**:\n   - 可能存在实验测定误差或数据偏离模型训练集，导致实际值与预测结果不一致。\n\n#### 步骤 5: 提供数据质量的综合评分和建议\n\n1. **数据质量评分**: 中等-偏低\n   - 输入数据在多个目标变量上的一致性较差，各模型的预测差异较大，需要确认数据采集过程中是否存在系统误差或偶然波动。\n\n2. **建议**:\n   - 检查实验条件和测量设备是否存在偏差。\n   - 通过多次重复实验以确认输入数据的稳定性。\n   - 考虑拓展模型训练集的代表性，提高模型在新样本预测中的可靠性。\n\n综上所述，当前的BMG数据点可能存在一定的可靠性问题，建议进一步验证和检查实验方法和数据处理过程，以提高可信性。",
    "anomaly_llm_output": "### 数据质量评估报告\n\n#### 1. 数据和模型预测结果\n\n**BMG实验数据**\n- Composition: Al20(CoCrCuFeMnNiTiV)80\n- Dmax(mm): 13.0\n- Tg(K): 423.0\n- Tx(K): 542.0\n- Tl(K): 812.0\n- Yield(MPa): 1634.0\n- Modulus(GPa): 60.0\n- Ε(%): 4.1\n\n**模型预测结果**\n- **Isolation Forest**：1（正常点）\n- **OneClass SVM**：1（正常点）\n- **Local Outlier Factor**：-1（异常点）\n- **Gaussian Mixture**：-171.58963505504232（较高异常概率）\n- **Autoencoder**：0.046534421261337375（较低重建误差）\n\n#### 2. 分析模型预测结果\n\n- **Isolation Forest** 和 **OneClass SVM** 均将该数据点标记为正常。这两种模型擅长于检测整体结构和边界是否存在明显差异。\n  \n- **Local Outlier Factor (LOF)** 认为该数据点为异常，表明该点的局部密度与其邻居有显著差异。LOF通常在检测局部异常上表现良好。\n\n- **Gaussian Mixture** 模型给出的负对数似然值较大，暗示数据点与已知的高斯分布差异较大，这是异常的一个信号。\n\n- **Autoencoder** 的重建误差较低，意味着在重构该数据点时表现尚可，通常表明数据点不是显著异常。\n\n#### 3. 评估数据点的可靠性和可信性\n\n综合模型的结果，数据点在**Isolation Forest**和**OneClass SVM**下显示为正常，这两个模型通常判断误差较少，适合全局评估。然而，**LOF**和**Gaussian Mixture**给出相异的结果，提示可能存在局部或分布偏离。**Autoencoder**的低重建误差提供了额外的置信。\n\n- **整体评估**：尽管存在分歧，但依赖多个模型的判断显示该点可能处于正常与异常的边界状态，局部和分布上不完全匹配。\n\n#### 4. 数据质量评分和整体评级\n\n- **数据点评分**：\n  - 普遍正常：4/5\n  - 局部异常迹象：1/5\n\n- **整体数据集质量评级**：良好-中等偏上（但需关注局部异常）\n\n#### 5. 建议\n\n1. **进一步核实实验步骤**：重新查看实验操作流程或仪器校准，以确认是否有误差源。\n2. **数据点复核**：考虑使用更多的局部分析模型，验证LOF和Gaussian Mixture判定。\n3. **数据集扩展**：引入更多类似成分比例的BMG数据点，校验该数据点的区域分布特性。\n\n#### 6. 结论\n\n尽管单独模型存在分歧，综合所有模型结果，当前BMG数据点的可靠性保持在可接受范围。建议在后续研究中持续监控类似特征数据点的表现。\n\n这份报告详细评估了当前BMG数据点，确保后续实验和分析的数据基础稳固，并对潜在异常进行了必要提示。",
    "outlier_llm_output": "### 数据质量评估报告\n\n#### 1. 实验数据整理\n\n- **Composition**: Al20(CoCrCuFeMnNiTiV)80\n- **Dmax(mm)**: 13.0\n- **Tg(K)**: 423.0\n- **Tx(K)**: 542.0\n- **Tl(K)**: 812.0\n- **yield(MPa)**: 1634.0\n- **Modulus(GPa)**: 60.0\n- **Ε(%)**: 4.1\n\n#### 2. 离群点检测算法输出结果\n\n- **BoxPlot模型**：\n  - 检测结果：True\n  - 说明：数据点被认为是离群点。\n\n- **ZScore模型**：\n  - 检测结果：False\n  - 说明：数据点在可接受范围内。\n\n- **DBSCAN模型**：\n  - 检测结果：True\n  - 说明：数据点被检测为噪声/离群点。\n\n- **KMeans模型**：\n  - 检测结果：False\n  - 说明：数据点与所属簇中心的距离在阈值内。\n\n#### 3. 离群点检测算法适用性分析\n\n每种算法采用不同的原理来检测离群点，适用于不同的数据集特性：\n\n- **BoxPlot模型**：适用于数据呈某种程度分布时，容易受到极端值影响。\n- **ZScore模型**：适用于较大数据集，前提是数据符合正态分布。\n- **DBSCAN模型**：适用于密度变化明显的数据集，非常适于检测密度差异。\n- **KMeans模型**：常用于已知簇数的数据，计算相对距离，易于受簇数影响。\n\n#### 4. 综合数据质量评估\n\n在评估数据质量时，需要平衡多个算法的结果：\n\n- **BoxPlot和DBSCAN**都检测到离群点，说明数据存在某些突出的异常特征。\n- **ZScore和KMeans**则认为数据在正常范围内，表明这些特点在某些分布类型中是合理的。\n\n这表明该数据点可能在某些情况下是异常的，可能需要进一步调查其异常原因或验证数据的准确性。\n\n#### 5. 数据质量评分\n\n基于以上分析，可以进行以下数据质量评分：\n\n- **可靠性**: 6/10\n  - 存在多个模型指出异常情况，导致数据可靠性略低。\n- **可信性**: 7/10\n  - 虽然两个模型认为是异常数据，但整体分布特性还需进一步验证，可信性一般。\n\n#### 6. 可能的数据问题和建议\n\n- **数据采集错误**：检查实验数据采集的准确性和仪器校准情况。\n- **数据范围验证**：确定该场合下数据范围的合理性，可能需要更大数据集。\n- **重复实验**：考虑重复实验以验证结果的稳定性。\n\n### 结论\n\n此次数据质量评估表明该BMG实验数据点可能为异常点，建议进一步验证和调查。另外，可以考虑增加更多的数据点进行更全面的分析，以提升评估的准确性。"
}
    async for chunk in llm_summary(None, llm_ana_results):
        print(chunk, end='', flush=True)
    

# python -m LLMClient.unit_test
if __name__ == "__main__":
    # llm_eval_rsp = asyncio.run(llm_eval_unit_test())
    # llm_regression_rsp = asyncio.run(llm_regression_unit_test())
    # llm_anomaly_rsp = asyncio.run(llm_anomaly_unit_test())
    # llm_outlier_rsp = asyncio.run(llm_outlier_unit_test())
    # results = {
    #     'llm_eval': llm_eval_rsp,
    #     'llm_regression': llm_regression_rsp,
    #     'llm_anomaly': llm_anomaly_rsp,
    #     'llm_outlier': llm_outlier_rsp
    # }
    # import json
    # print(json.dumps(results, indent=4, ensure_ascii=False))
    asyncio.run(llm_summary_unit_test())