import json
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import logging
from Models.models import Model
from Regression.regressioner import Regressioner

logger = logging.getLogger(__name__)


class RuleJudger(Model):
    def __init__(self, data_path, drop_columns, target_columns, regressioner: Regressioner):
        super().__init__(data_path, drop_columns, target_columns)
        self.drop_columns = drop_columns
        self.original_data = pd.read_excel(data_path)
        self.thresholds = self.get_column_thresholds()
        self.feature_columns = list(self.data.data.drop(columns=target_columns).columns)
        self.regressioner = regressioner
        logger.info("RuleJudger initialized.")
        
    @staticmethod
    def parse_elements(composition):
        pattern = r"\(([A-Za-z\d.]+)\)([\d.]+)|([A-Z][a-z]*)([\d.]+)?"
        matches = re.findall(pattern, composition)
        elements = {}
        for match in matches:
            if match[0]:  # 如果有括号
                sub_scale_factor = sum(float(x[1]) if x[1] else 1 for x in re.findall(r"([A-Z][a-z]*)([\d.]+)?", match[0]))
                scale_factor = float(match[1]) / sub_scale_factor
                sub_matches = re.findall(r"([A-Z][a-z]*)([\d.]+)?", match[0])
                for sub_match in sub_matches:
                    if sub_match[0] not in elements:
                        elements[sub_match[0]] = float(sub_match[1]) * scale_factor if sub_match[1] else scale_factor
            elif match[2]:  # 如果没有括号
                if match[2] not in elements:
                    elements[match[2]] = float(match[3]) if match[3] else 1
        sorted_elements = {key: value for key, value in sorted(elements.items(), key=lambda item: item[1], reverse=True)}
        element_strings = [f"{element}{round(value,2)}" for element, value in sorted_elements.items()]
        chem = "".join(element_strings)
        return elements, chem
    
    def get_column_thresholds(self):
        thresholds = {}
        for column in self.data.data.columns:
            thresholds[column] = int(self.data.data[column].min()), int(self.data.data[column].max())
        return thresholds 
    
    def judge_composition(self, elements: dict, chem: str) -> Tuple[str, str]:
        check_results = []
        if sum(elements.values()) != 100:
            check_results.append({'error': f'❌ BMGs的元素总和不等于100，请检查或使用下面修正的成分，目前解析的BMGs成分为{chem}，建议删除中括号等复杂计算方式，直接输入元素百分比，如`Al20Cu80`。'})
            total = sum(elements.values())
            elements = {key: round(100 * value / total, 2) for key, value in elements.items()}
            check_results.append({'warning': f'⚠️ BMGs的元素总和不等于100，已启动自动修正，修正后的BMGs成分为{elements}。'})
        else:
            check_results.append({'info': f'✅ BMGs的元素总和等于100，BMGs成分为{elements}。'})
        return check_results, elements, chem

    def get_composition_features(self, elements: dict) -> np.ndarray:
        x = np.zeros(len(self.feature_columns))
        for element, value in elements.items():
            if element in self.feature_columns:
                x[self.feature_columns.index(element)] = value
        return x
        
    def judge_targets(self, input_point: dict, x: np.ndarray) -> List[dict]:
        check_results = []
        for column in self.target_columns:
            # 数据不存在或者为空或者不是数字, 数字是浮点数
            try:
                input_point[column] = float(input_point[column])
                logger.info(f"column: {column} is not numeric")
            except:
                input_point[column] = ""
            if column not in input_point or not input_point[column]:
                predicted_value = round(self.regressioner.predict(x, column), 2)
                check_results.append({'warning': f'⚠️ {column}的数据不存在或者为空或者不是数字，已启动自动预测，预测值为{predicted_value}。'})
                input_point[f"{column}(predicted)"] = predicted_value
            else:
                value = round(float(input_point[column]), 2)
                min_value, max_value = self.thresholds[column]
                if value < min_value:
                    check_results.append({'warinig': f'⚠️ {column}的数据不在数据集的合理范围内，{column}为{value}，数据集中的最小值为{min_value}，请检查。'})
                elif value > max_value:
                    check_results.append({'warinig': f'⚠️ {column}的数据不在数据集的合理范围内，{column}为{value}，数据集中的最大值为{max_value}，请检查。'})
                else:
                    check_results.append({'info': f'✅ {column}的数据在合理范围内，{column}为{value}。'})
                
        return check_results, input_point
    
    def find_similar_points(self, x):
        check_results = []
        all_features = self.data.data.drop(columns=self.target_columns).to_numpy()
        distances = np.linalg.norm(all_features - x, axis=1)
        within_threshold_indices = np.where(distances <= 10)[0]
        similar_points = [(index, distances[index]) for index in within_threshold_indices]
        similar_points = sorted(similar_points, key=lambda point: point[1])
        indexes = [point[0] for point in similar_points]
        if len(indexes) > 0:
            similar_bmgs = "📌 在数据集中找到了以下相似的BMGs："
            for index in indexes:
                BMGs = self.original_data.loc[index, 'BMGs']
                similar_bmgs += f"\n    - BMGs: {BMGs}"
                for target_column in self.target_columns:
                    if not np.isnan(self.original_data.loc[index, target_column]):
                        similar_bmgs += f" {target_column}: {self.original_data.loc[index, target_column]}"
            check_results.append({'info': similar_bmgs})
        else:
            similar_bmgs = None
            check_results.append({'warning': f'⚠️ 没有找到与输入点相似的点。'})
        return check_results, similar_bmgs
    
    @staticmethod
    def format_messages(messages):
        md_string = ""
        
        for msg in messages:
            # 处理 info 信息
            if 'info' in msg:
                md_string += f"🟢 **Info**: {msg['info']}\n\n"
            
            # 处理 error 信息
            if 'error' in msg:
                md_string += f"🔴 **Error**: {msg['error']}\n\n"
            
            # 处理 warning 信息
            if 'warning' in msg:
                md_string += f"🟡 **Warning**: {msg['warning']}\n\n"
        
        return md_string
    
    def construct_full_x_y(self, x, input_point):
        results = {}
        for target_name in self.target_columns:
            if f"{target_name}(predicted)" in input_point:
                results[target_name] = input_point[f"{target_name}(predicted)"]
            else:
                results[target_name] = input_point[target_name]
                
        for column_name, value in zip(self.feature_columns, x):
            results[column_name] = value
        all_columns = self.original_data.drop(columns=self.drop_columns).columns
        y = np.array([results[column] for column in all_columns if column in self.target_columns])
        full_x_y = np.array([results[column] for column in all_columns])
        return full_x_y, y
    
    def judge(self, input_point):
        """
        input_point:
        {
            "Composition": "Al20(CoCrCuFeMnNiTiV)80",
            "Dmax(mm)": "13",
            "Tg(K)": "345",
            "Tx(K)": "467",
            "Tl(K)": "831",
            "yield(MPa)": "1234",
            "Modulus(GPa)": "164",
            "Ε(%)": "23"
        }
        """
        all_check_results = []
        elements, chem = self.parse_elements(input_point["Composition"])
        composition_check_results, elements, chem = self.judge_composition(elements, chem)
        x = self.get_composition_features(elements)
        all_check_results.extend(composition_check_results)
        targets_check_results, input_point = self.judge_targets(input_point, x)
        all_check_results.extend(targets_check_results)
        similar_points_results, similar_bmgs = self.find_similar_points(x)
        all_check_results.extend(similar_points_results)
        md_str = self.format_messages(all_check_results)
        full_x_y, y = self.construct_full_x_y(x, input_point)
        return md_str, input_point, x, y, full_x_y, similar_bmgs
    
def unit_test():
    data_path = '../data/ALL_data_grouped_processed.xlsx'  # Replace with your file path
    drop_columns = ['BMGs', "Chemical composition", 'cls_label']
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus(GPa)', 'Ε(%)']
    regressioner = Regressioner(data_path, drop_columns, target_columns)
    judger = RuleJudger(data_path, drop_columns, target_columns, regressioner)
    test_point = {
        "Composition": "Al20(CoCrCuFeMnNiTiV)80",
        "Dmax(mm)": "xx",
        "Tg(K)": "xxx",
        "Tx(K)": "xxx",
        "Tl(K)": "xxx",
        "yield(MPa)": "1465",
        "Modulus(GPa)": "190",
        "Ε(%)": "2.35"
    }
    md_str, input_point, x, y, full_x_y, similiar_bmg = judger.judge(test_point)
    print(md_str)
    print(input_point)
    print(x)
    print(y)
    print(full_x_y)
    print(similiar_bmg)

# python -m RuleJudge.judger
if __name__ == "__main__":
    unit_test()