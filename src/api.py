import os
import uuid
from asyncio import sleep
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from Anomaly.anomaly_detector import AnomalyDetector
from config import logging
from DimReduction.reducer import DimensionalityReducer
from LLMClient.llm_ana import (llm_anomaly, llm_eval, llm_outlier,
                               llm_regression, llm_summary)
from Outlier.outlier_detector import OutlierDetector
from Regression.regressioner import Regressioner
from RuleJudge.judger import RuleJudger
from io import StringIO

logger = logging.getLogger(__name__)


app = FastAPI()
# 挂载静态文件目录
if not os.path.exists("static"):
    os.mkdir("./static")
app.mount("/static", StaticFiles(directory="static"), name="static")


global_variable = None

data_path = '../data/ALL_data_grouped_processed.xlsx'  # Replace with your file path
data_predict_path = '../data/ALL_data_grouped_processed_predict.xlsx'  # Replace with your file path
drop_columns = ['BMGs', "Chemical composition", 'cls_label']
target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus(GPa)', 'Ε(%)']
regressioner = Regressioner(data_path, drop_columns, target_columns)
rule_judger = RuleJudger(data_path, drop_columns, target_columns, regressioner)
ruducer = DimensionalityReducer(data_predict_path, drop_columns, target_columns)
anomaly_detector = AnomalyDetector(data_predict_path, drop_columns, target_columns)
outlier_detector = OutlierDetector(data_predict_path, drop_columns, target_columns)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global global_variable
    # 初始化全局变量
    global_variable = {}
    logger.info("Global variable initialized")
    
    yield
    global_variable = None
    logger.info("Global variable cleaned")

app.router.lifespan_context = lifespan
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_params(input_data: dict):
    logger.info(f"Params received: {input_data}")
    global_variable['input_point'] = input_data['point']
    global_variable['model_name'] = input_data.get('model_name', 'gpt-4o')
    global_variable['reduce'] = input_data.get('reduce', ['PCA', 't-SNE'])
    global_variable['anomaly'] = input_data.get('anomaly', ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 'GaussianMixture', 'Autoencoder'])
    global_variable['outlier'] = input_data.get('outlier', ['BoxPlot', 'ZScore', 'DBSCAN', 'KMeans'])
    global_variable['regression'] = input_data.get('regression', ['ElasticNet', 'SVR', 'RandomForestRegressor', 'KNeighborsRegressor', 'XGBRegressor'])
    logger.info(f"Params parsed: {global_variable}")


async def process_llm_response(llm_func, input_point, results, model_name, plt_key=None, plts=None, result_key=None):
    # 创建一个用于保存输出的缓冲区
    output_buffer = StringIO()

    # 异步生成器函数，捕获和发送数据流
    async def event_stream():
        async for item in llm_func(input_point, results, model_name=model_name):
            # 将数据流发送给前端
            yield item
            # 同时将数据写入内存缓冲区
            output_buffer.write(item)

    # 如果有图像数据，保存到 global_variable
    if plt_key and plts:
        global_variable[plt_key] = plts

    # 处理流式响应
    response = StreamingResponse(event_stream(), media_type="text/event-stream")

    # 在响应结束后保存结果
    if result_key:
        global_variable[result_key] = output_buffer.getvalue()
        logger.info(f"LLM output saved to {result_key}")

    return response

def save_results(result_key, results, logger, log_msg):
    global_variable[result_key] = results
    logger.info(log_msg)

async def emulate_stream(md_str):
    for i in md_str:
        yield i
        await sleep(0.03)

@app.post("/get_rule_judge")
async def get_rule_judge(request: Request):
    input_data = await request.json()
    parse_params(input_data=input_data)
    input_point = global_variable['input_point']
    md_str, input_point, x, y, full_x_y, similar_bmgs = rule_judger.judge(input_point)
    global_variable['input_point'] = input_point
    global_variable['x'] = x
    global_variable['y'] = y
    global_variable['full_x_y'] = full_x_y
    global_variable['similar_bmgs'] = similar_bmgs
    global_variable['model_name'] = input_data['model_name']
    return StreamingResponse(emulate_stream(md_str), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

@app.post("/get_llm_eval")
async def get_llm_eval(request: Request):
    similar_bmgs = global_variable['similar_bmgs']
    input_point = global_variable['input_point']
    
    return await process_llm_response(
        llm_func=llm_eval,
        input_point=input_point,
        results=similar_bmgs,
        model_name=global_variable['model_name'],
        result_key='eval_llm_output'
    )

@app.post("/get_llm_regression")
async def get_llm_regression(request: Request):
    x = global_variable['x']
    regression_results = regressioner.regression(x, global_variable['regression'])
    
    save_results(
        result_key='regression_results',
        results=regression_results,
        logger=logger,
        log_msg=f"Regression results: {regression_results}"
    )
    
    return await process_llm_response(
        llm_func=llm_regression,
        input_point=global_variable['input_point'],
        results=regression_results,
        model_name=global_variable['model_name'],
        result_key='regression_llm_output'
    )

@app.post("/get_llm_anomaly")
async def get_llm_anomaly(request: Request):
    full_x_y = global_variable['full_x_y']
    anomaly_results, anomaly_plts = anomaly_detector.anomaly_detect(full_x_y, global_variable['anomaly'])
    
    save_results(
        result_key='anomaly_results',
        results=anomaly_results,
        logger=logger,
        log_msg=f"Anomaly results: {anomaly_results}"
    )
    
    return await process_llm_response(
        llm_func=llm_anomaly,
        input_point=global_variable['input_point'],
        results=anomaly_results,
        model_name=global_variable['model_name'],
        plt_key='anomaly_plts',
        plts=anomaly_plts,
        result_key='anomaly_llm_output'
    )

@app.post("/get_llm_outlier")
async def get_llm_outlier(request: Request):
    full_x_y = global_variable['full_x_y']
    outlier_results, outlier_plts = outlier_detector.outlier_detect(full_x_y, global_variable['outlier'])
    
    save_results(
        result_key='outlier_results',
        results=outlier_results,
        logger=logger,
        log_msg=f"Outlier results: {outlier_results}"
    )
    
    return await process_llm_response(
        llm_func=llm_outlier,
        input_point=global_variable['input_point'],
        results=outlier_results,
        model_name=global_variable['model_name'],
        plt_key='outlier_plts',
        plts=outlier_plts,
        result_key='outlier_llm_output'
    )

@app.post("/get_dim_reduction")
async def get_dim_reduction(request: Request):
    x = global_variable['x']
    y = global_variable['y']
    reduce_plts = ruducer.dim_reduce(x, y, global_variable['reduce'])
    if reduce_plts:
        global_variable['reduce_plts'] = reduce_plts
    return Response(status_code=200)

@app.post("/get_plts")
async def get_plots():
    plt_objs = []
    if 'reduce_plts' in global_variable:
        plt_objs.extend(list(global_variable['reduce_plts'].values()))
    if 'anomaly_plts' in global_variable:
        plt_objs.extend(list(global_variable['anomaly_plts'].values()))
    if 'outlier_plts' in global_variable:
        plt_objs.extend(list(global_variable['outlier_plts'].values()))
    urls = []
    for plt_obj in plt_objs:
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join("static", filename)
        plt_obj.savefig(filepath)
        url = f"http://127.0.0.1:8000/static/{filename}"
        urls.append(url)
    return urls
    

@app.post("/get-summary")
async def summary(request: Request):
    llm_ana_results = {}
    for key in ['eval_llm_output', 'regression_llm_output', 'anomaly_llm_output', 'outlier_llm_output']:
        if key in global_variable:
            llm_ana_results[key] = global_variable[key]
    return await process_llm_response(
        llm_func=llm_summary,
        input_point=None,
        results=llm_ana_results,
        model_name=global_variable['model_name'],
        result_key='summary_llm_output'
    )
    
# python -m uvicorn api:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)