from LLMClient.llm_call import get_rsp_from_GPT
from LLMClient.prompts import *


def construct_data_str(input_point):
    data_str = f"Composition: {input_point['Composition']} "
    for key in ["Dmax(mm)", "Tg(K)", "Tx(K)", "Tl(K)", "yield(MPa)", "Modulus(GPa)", "Ε(%)"]:
        if not input_point[key]:
            continue
        try:
            input_point[key] = float(input_point[key])
            data_str += f"{key}: {input_point[key]} "
        except:
            pass
    return data_str
    

def llm_eval(input_point, similar_bmgs):
    sys_prompt = EvalueSystem
    data_str = construct_data_str(input_point)
    user_prompt = EvalueUser.format(rule=Rules, similar_bmgs=similar_bmgs, data=data_str)
    for chunk in get_rsp_from_GPT(sys_prompt, user_prompt):
        yield chunk

def llm_regression(input_point, regression_results):
    regression_results_str = []
    for model, content in regression_results.items():
        regression_results_str.append(f"{model} 模型：")
        regression_results_str.append(f"模型描述：{content['model_description']}")
        regression_results_str.append(f"模型预测结果：Tg(K) = {content['targets_results']['Tg(K)']}, "
                      f"Tx(K) = {content['targets_results']['Tx(K)']}, "
                      f"Tl(K) = {content['targets_results']['Tl(K)']}, "
                      f"Dmax(mm) = {content['targets_results']['Dmax(mm)']}, "
                      f"屈服强度(MPa) = {content['targets_results']['yield(MPa)']}, "
                      f"杨氏模量(GPa) = {content['targets_results']['Modulus(GPa)']}, "
                      f"断裂延伸率(%) = {content['targets_results']['Ε(%)']}"
                      "\n")
    data_str = construct_data_str(input_point)
    regression_results_str  = "\n".join(regression_results_str)
    sys_prompt = RegressionSystem
    user_prompt = RegressionUser.format(data=data_str, regression_results=regression_results_str)
    for chunk in get_rsp_from_GPT(sys_prompt, user_prompt):
        yield chunk

def llm_anomaly(input_point, anomaly_results):
    anomaly_result_str = []
    for model, content in anomaly_results.items():
        anomaly_result_str.append(f"{model} 模型：")
        anomaly_result_str.append(f"模型描述：{content['description']}")
        anomaly_result_str.append(f"检测结果：{content['result']}\n")
    anomaly_result_str = "\n".join(anomaly_result_str)
    sys_prompt = AnomalySystem
    data_str = construct_data_str(input_point)
    user_prompt = AnomalyUser.format(data=data_str, anomaly_results=anomaly_result_str)
    for chunk in get_rsp_from_GPT(sys_prompt, user_prompt):
        yield chunk
        
def llm_outlier(input_point, outlier_results):
    outlier_result_str = []
    for model, content in outlier_results.items():
        outlier_result_str.append(f"{model} 模型：")
        outlier_result_str.append(f"模型描述：{content['description']}")
        outlier_result_str.append(f"检测结果：{content['result']}\n")
    outlier_result_str = "\n".join(outlier_result_str)
    sys_prompt = OutlierSystem
    data_str = construct_data_str(input_point)
    user_prompt = OutlierUser.format(data=data_str, outlier_results=outlier_result_str)
    for chunk in get_rsp_from_GPT(sys_prompt, user_prompt):
        yield chunk