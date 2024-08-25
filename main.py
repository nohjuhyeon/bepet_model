from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,            
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static 파일 경로 설정
app.mount("/images", StaticFiles(directory="resources/images/"), name="static_images")
app.mount("/inputs", StaticFiles(directory="resources/inputs/"), name="input_images")
app.mount("/outputs", StaticFiles(directory="resources/outputs/"), name="output_images")
app.mount("/models", StaticFiles(directory="resources/models/"), name="models")

templates = Jinja2Templates(directory="templates/")    

# YOLO 모델 로드
try:
    yolo_model = YOLO('resources/models/yolov8n.pt')
    print('YOLO 모델을 성공적으로 로드했습니다.')
except:
    print('YOLO 모델을 성공적으로 로드하지 못했습니다.')

@app.get("/")                     
async def main_get(request:Request):
    return templates.TemplateResponse("main.html", {'request': request})

@app.post("/result")                     
async def result_post(request: Request, user_img: UploadFile = File(...)):
    # 업로드된 파일을 저장할 경로 설정
    upload_dir = "resources/inputs"
    os.makedirs(upload_dir, exist_ok=True)
    file_location = f"{upload_dir}/{user_img.filename}"

    # 파일 저장
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(user_img.file, file_object)
    
    # YOLO 모델을 사용한 객체 탐지 수행
    img = Image.open(file_location)
    results = yolo_model.predict(img)

    # 결과 이미지에 박스 그리기
    for result in results:
        # 박스 탐지가 안될 경우 원본 이미지로 result.orgi_img 사용
        if len(result)==0:
            orig_img = result.orig_img
        # 박스 탐시가 된 경우 원본 이미지로 result[1].orig_img 사용
        else:
            orig_img = result[1].orig_img  
        boxes = result.boxes  # 탐지된 객체의 박스 정보
        names = result.names  # 클래스 이름들
        
        # 원본 이미지 시각화
        plt.figure(figsize=(10, 10))
        plt.imshow(orig_img)
        plt.axis('off')

        ax = plt.gca()
        ax.set_axis_off()
        # 이미지에 박스 및 클래스 이름들 표시
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            class_name = names[class_id]
            box_line = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(box_line)
            plt.text(x1, y1, class_name, color='red', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

        # resources/outputs/ 에 분석 결과 이미지 저장
        output_filename = f"resources/outputs/output_{user_img.filename}.png"
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 분석 결과(추후 변경 예정)
    output_path = f'/outputs/output_{user_img.filename}.png'
    disease_name = '백내장'
    disease_cause = '백내장 원인에는 품종 유전, 당뇨병, 의상, 포도상구균, 진행성 망막위축증, 망막박리 등이 있습니다.'
    disease_symptom = '백내장 증상에는 동공이 흐릿하거나 밤눈이 어둡거나 눈 색깔의 변화 등이 있습니다.'
    disease_treatment = '치료법은 약물치료, 수술 등이 있습니다.'

    # 결과 페이지로 전달
    return templates.TemplateResponse("result.html", {
        'request': request,
        'input_path': f"/inputs/{user_img.filename}",
        'output_path': output_path,
        'disease_name': disease_name,
        'disease_cause': disease_cause,
        'disease_symptom': disease_symptom,
        'disease_treatment': disease_treatment
    })
