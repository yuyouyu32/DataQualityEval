import os
import uuid
from asyncio import sleep
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from Anomaly.anomaly_detector import AnomalyDetector
from config import logging
from DimReduction.reducer import DimensionalityReducer
from LLMClient.llm_ana import (llm_anomaly, llm_eval, llm_outlier,
                               llm_regression)
from Outlier.outlier_detector import OutlierDetector
from Regression.regressioner import Regressioner
from RuleJudge.judger import RuleJudger

logger = logging.getLogger(__name__)


app = FastAPI()

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
    global_variable['input_point'] = input_data['point']
    global_variable['model_name'] = input_data.get('model_name', 'GPT-4o')
    global_variable['reduce'] = input_data.get('reduce', ['PCA', 't-SNE'])
    global_variable['anomaly'] = input_data.get('anomaly', ['IsolationForest', 'OneClassSVM', 'LocalOutlierFactor', 'GaussianMixture', 'Autoencoder'])
    global_variable['outlier'] = input_data.get('outlier', ['BoxPlot', 'ZScore', 'DBSCAN', 'KMeans'])
    global_variable['regression'] = input_data.get('regression', ['ElasticNet', 'SVR', 'RandomForestRegressor', 'KNeighborsRegressor', 'XGBRegressor'])
    logger.info(f"Params parsed: {global_variable}")


async def emulate_stream(md_str):
    for i in md_str:
        yield i
        await sleep(0.01)

@app.get("/get_rule_judge")
async def get_rule_judge(request: Request):
    input_data = await request.json()
    input_point = input_data['Point']
    md_str, input_point, x, y, full_x_y, similar_bmgs = rule_judger.judge(input_point)
    global_variable['input_point'] = input_point
    global_variable['x'] = x
    global_variable['y'] = y
    global_variable['full_x_y'] = full_x_y
    global_variable['similar_bmgs'] = similar_bmgs
    global_variable['model_name'] = input_data['Agent']
    return StreamingResponse(emulate_stream(md_str), media_type="text/event-stream")


@app.get("/get_llm_eval")
async def get_llm_eval(request: Request):
    similar_bmgs = global_variable['similar_bmgs']
    input_point = global_variable['input_point']
    return StreamingResponse(llm_eval(input_point, similar_bmgs, model_name=global_variable['model_name']), media_type="text/event-stream")
    
@app.get("/get_llm_regression")
async def get_llm_regression(request: Request):
    x = global_variable['x']
    regression_results = regressioner.regression(x, global_variable['regression'])
    logger.info(f"Regression results: {regression_results}")
    global_variable['regression_results'] = regression_results
    return StreamingResponse(llm_regression(global_variable['input_point'], regression_results, model_name=global_variable['model_name']), media_type="text/event-stream")

@app.get("/get_llm_anomaly")
async def get_llm_anomaly(request: Request):
    full_x_y = global_variable['full_x_y']
    anomaly_results, anomaly_plts = anomaly_detector.anomaly_detect(full_x_y, global_variable['anomaly'])
    logger.info(f"Anomaly results: {anomaly_results}")
    global_variable['anomaly_results'] = anomaly_results
    if anomaly_plts:
        global_variable['anomaly_plts'] = anomaly_plts
    return StreamingResponse(llm_anomaly(global_variable['input_point'], anomaly_results, model_name=global_variable['model_name']), media_type="text/event-stream")

@app.get("/get_llm_outlier")
async def get_llm_outlier(request: Request):
    full_x_y = global_variable['full_x_y']
    outlier_results, outlier_plts = outlier_detector.outlier_detect(full_x_y, global_variable['outlier'])
    logger.info(f"Outlier results: {outlier_results}")
    global_variable['outlier_results'] = outlier_results
    if outlier_plts:
        global_variable['outlier_plts'] = outlier_plts
    return StreamingResponse(llm_outlier(global_variable['input_point'], outlier_results, model_name=global_variable['model_name']), media_type="text/event-stream")

@app.get("/get_dim_reduction")
async def get_dim_reduction(request: Request):
    x = global_variable['x']
    y = global_variable['y']
    reduce_plts = ruducer.dim_reduce(x, y, global_variable['reduce'])
    if reduce_plts:
        global_variable['reduce_plts'] = reduce_plts
    return Response(status_code=200)

@app.get("/get_plts")
async def get_plots():
    
    # 获取所有的图像对象
    plt_objs = []
    if 'reduce_plts' in global_variable:
        plt_objs.extend(global_variable['reduce_plts'])
    if 'anomaly_plts' in global_variable:
        plt_objs.extend(global_variable['anomaly_plts'])
    if 'outlier_plts' in global_variable:
        plt_objs.extend(global_variable['outlier_plts'])
    urls = []
    for plt_obj in plt_objs:
        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join("static", filename)
        plt_obj.savefig(filepath)
        url = f"/static/{filename}"
        urls.append(url)

    return {"plot_urls": urls}
    

@app.post("/get-summary")
async def summary():
    return StreamingResponse(emulate_stream("Summary"*20), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)