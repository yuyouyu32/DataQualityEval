import concurrent.futures
from asyncio import sleep
from concurrent.futures import ThreadPoolExecutor

from LLMClient.llm_call import get_rsp_from_GPT
from LLMClient.prompts import *
from config import logging

logger = logging.getLogger(__name__)


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
    

async def llm_eval(input_point, similar_bmgs, model_name: str='gpt-4o'):
    sys_prompt = EvalueSystem
    data_str = construct_data_str(input_point)
    user_prompt = EvalueUser.format(rule=Rules, similar_bmgs=similar_bmgs, data=data_str)
    for chunk in get_rsp_from_GPT(sys_prompt, user_prompt, model_name=model_name):
        yield chunk
        await sleep(0.01)

async def llm_regression(input_point, regression_results, model_name: str='gpt-4o'):
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
    for chunk in get_rsp_from_GPT(sys_prompt, user_prompt, model_name=model_name):
        yield chunk
        await sleep(0.01)
         

async def llm_anomaly(input_point, anomaly_results, model_name: str='gpt-4o'):
    anomaly_result_str = []
    for model, content in anomaly_results.items():
        anomaly_result_str.append(f"{model} 模型：")
        anomaly_result_str.append(f"模型描述：{content['description']}")
        anomaly_result_str.append(f"检测结果：{content['result']}\n")
    anomaly_result_str = "\n".join(anomaly_result_str)
    sys_prompt = AnomalySystem
    data_str = construct_data_str(input_point)
    user_prompt = AnomalyUser.format(data=data_str, anomaly_results=anomaly_result_str)
    for chunk in get_rsp_from_GPT(sys_prompt, user_prompt, model_name=model_name):
        yield chunk
        await sleep(0.01)
        
async def llm_outlier(input_point, outlier_results, model_name: str='gpt-4o'):
    outlier_result_str = []
    for model, content in outlier_results.items():
        outlier_result_str.append(f"{model} 模型：")
        outlier_result_str.append(f"模型描述：{content['description']}")
        outlier_result_str.append(f"检测结果：{content['result']}\n")
    outlier_result_str = "\n".join(outlier_result_str)
    sys_prompt = OutlierSystem
    data_str = construct_data_str(input_point)
    user_prompt = OutlierUser.format(data=data_str, outlier_results=outlier_result_str)
    for chunk in get_rsp_from_GPT(sys_prompt, user_prompt, model_name=model_name):
        yield chunk
        await sleep(0.01)
        
async def llm_summary(input_point, llm_ana_results, model_name: str = 'gpt-4o'):
    summary_report = {}
    system_prompt = "你是一位经验丰富的数据质量分析师，从各种数据评估结果中提取关键信息，形成一个精简有效的总结报告。"
    # 定义多线程执行的函数
    def process_summary(key, full_report):
        user_prompt = chunk_summary_prompt[key].format(full_report=full_report)
        print(key)
        print(f"User prompt: {user_prompt}")
        return key, get_rsp_from_GPT(system_prompt, user_prompt, model_name=model_name, stream=False)
    
    # 使用多线程执行每个报告的处理
    with ThreadPoolExecutor(max_workers=len(llm_ana_results)) as executor:
        # 提交任务并收集结果
        future_to_key = {executor.submit(process_summary, key, full_report): key for key, full_report in llm_ana_results.items()}
        
        for future in concurrent.futures.as_completed(future_to_key):
            key, summary = future.result()
            summary_report[key] = summary
    logger.info(f"Summary report: {summary_report}")
    # 生成最终的总结报告
    summary_system_prompt = SummarySystem
    summary_user_prompt = SummaryUser.format(**summary_report)
    
    # 流式生成最终的总结报告
    for chunk in get_rsp_from_GPT(summary_system_prompt, summary_user_prompt, model_name=model_name):
        yield chunk
        await sleep(0.01)