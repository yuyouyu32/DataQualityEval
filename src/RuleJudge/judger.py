import numpy as np
import re

from config import logging
from Models.models import Model

logger = logging.getLogger(__name__)


class RuleJudger(Model):
    def __init__(self, data_path, drop_columns, target_columns):
        super().__init__(data_path, drop_columns, target_columns)
        self.thresholds = self.get_column_thresholds()
        self.feature_columns = list(self.data.data.drop(columns=target_columns).columns)
    
    def get_column_thresholds(self):
        thresholds = {}
        for column in self.data.data.columns:
            thresholds[column] = int(self.data.data[column].min()), int(self.data.data[column].max())
        return thresholds
        
    def parse_elements(composition):
        pattern = r"\[([A-Za-z\d.\(\)]+)]([\d.]+)|\(([A-Za-z\d.]+)\)([\d.]+)|([A-Z][a-z]*)([\d.]+)?"
        matches = re.findall(pattern, composition)
        elements = {}

        for match in matches:
            if match[0]:  # 如果有方括号
                bracket_scale_factor = float(match[1])
                bracket_content = match[0]
                bracket_matches = re.findall(pattern, bracket_content)

                for bracket_match in bracket_matches:
                    if bracket_match[2]:  # 如果有括号
                        paren_scale_factor = float(bracket_match[3])
                        paren_content = bracket_match[2]
                        paren_matches = re.findall(r"([A-Z][a-z]*)([\d.]+)?", paren_content)

                        for element, weight in paren_matches:
                            scaled_weight = float(weight) * paren_scale_factor * bracket_scale_factor if weight else paren_scale_factor * bracket_scale_factor

                            if element not in elements:
                                elements[element] = scaled_weight
                            else:
                                elements[element] += scaled_weight
                            
                    elif bracket_match[4]:  # 如果没有括号
                        element, weight = bracket_match[4], bracket_match[5]
                        scaled_weight = float(weight) * bracket_scale_factor if weight else bracket_scale_factor

                        if element not in elements:
                            elements[element] = scaled_weight
                        else:
                            elements[element] += scaled_weight

            elif match[2]:  # 如果有括号，但没有方括号
                paren_scale_factor = float(match[3])
                paren_content = match[2]
                paren_matches = re.findall(r"([A-Z][a-z]*)([\d.]+)?", paren_content)

                for element, weight in paren_matches:
                    scaled_weight = float(weight) * paren_scale_factor if weight else paren_scale_factor

                    if element not in elements:
                        elements[element] = scaled_weight
                    else:
                        elements[element] += scaled_weight

            elif match[4]:  # 如果没有括号和方括号
                element, weight = match[4], match[5]
                scaled_weight = float(weight) if weight else 1

                if element not in elements:
                    elements[element] = scaled_weight
                else:
                    elements[element] += scaled_weight

        sorted_elements = {key: value for key, value in sorted(elements.items(), key=lambda item: item[1], reverse=True)}
        element_strings = [f"{element}{round(value,2)}" for element, value in sorted_elements.items()]
        chem = "".join(element_strings)
        return elements, chem
    
    
    def judge(self, features):
        """
        Judge the input features based on the rules.

        Parameters:
            features (dict): The input features.

        Returns:
            dict: The judgment results.
        """
        results = {}
        for target_column in self.target_columns:
            results[target_column] = self.judge_target(features, target_column)
        return results

def unit_test():
    data_path = '../data/ALL_data_grouped_processed.xlsx'  # Replace with your file path
    drop_columns = ['BMGs', "Chemical composition", 'cls_label']
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus(GPa)', 'Ε(%)']
    judget = RuleJudger(data_path, drop_columns, target_columns)
    

# python -m RuleJudge.judger
if __name__ == "__main__":
    unit_test()