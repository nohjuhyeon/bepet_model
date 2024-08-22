from fastapi import FastAPI               
app = FastAPI()

from fastapi import Request                                
from fastapi.templating import Jinja2Templates              

from fastapi.middleware.cors import CORSMiddleware             
app.add_middleware(
    CORSMiddleware,            
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
# # url 경로(url에서 입력해야 하는 주소), 자원 물리 경로(directory; 실제 경로), 프로그래밍 측면(key의 이름 지정)
app.mount("/css", StaticFiles(directory="resources\\css\\"), name="static_css")
app.mount("/images", StaticFiles(directory="resources\\images\\"), name="static_images")

templates = Jinja2Templates(directory="templates/")    

# 메인 페이지로 이동
@app.get("/")                     
async def main_get(request:Request):
    print(dict(request._query_params))
    return templates.TemplateResponse("main.html",{'request':request})

@app.post("/")                      
async def main_post(request:Request):
    await request.form()
    print(dict(await request.form()))
    return templates.TemplateResponse("main.html",{'request':request})

# 메인 페이지로 이동
@app.get("/result")                     
async def main_get(request:Request):
    print(dict(request._query_params))
    return templates.TemplateResponse("result.html",{'request':request})

@app.post("/result")                      
async def main_post(request:Request):
    await request.form()
    print(dict(await request.form()))
    return templates.TemplateResponse("result.html",{'request':request})
