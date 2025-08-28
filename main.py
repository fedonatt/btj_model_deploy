from fastapi import FastAPI
from dotenv import load_dotenv
from api.predict.views import predict_router

load_dotenv(dotenv_path='.dev.env')

app = FastAPI(title="btj-academy- model deployment(iris)",)

app.include_router(predict_router)

@app.get("/")
def read_root():
    return {"Hello": "World"}

