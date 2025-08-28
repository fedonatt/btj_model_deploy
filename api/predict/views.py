from fastapi import APIRouter, HTTPException, status
from api.predict.service import Predict
from api.predict.schemas import PredictionParams, PredictionResult, ErrorResponse

predict_router = APIRouter()
tag = ["Predict"]

error_responses = {
    404: {"model": ErrorResponse, "description": "Model/Scaler file not found"},
    422: {"model": ErrorResponse, "description": "Invalid input / shape"},
    503: {"model": ErrorResponse, "description": "Prediction pipeline failed"},
    504: {"model": ErrorResponse, "description": "Upstream timeout"},
    500: {"model": ErrorResponse, "description": "Unexpected error"},
}

@predict_router.post(
    "/predict", 
    tags=tag,
    response_model=PredictionResult,
    responses=error_responses
)
def predict_route(data: PredictionParams) -> PredictionResult:
    try:
        pred = Predict(params=data).predict()
        return PredictionResult(**pred)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": str(e), "code": 404, "hint": "file model atau scaler tidak ditemukan"}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(e), "code": 422, "hint": "nilai input tidak sesuai"}
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"message": str(e), "code": 504, "hint": "upstream service timeout"}
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": str(e), "code": 503, "hint": "pipeline prediksi gagal dijalankan"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "code": 500, "hint": "unexpected error"}
        )

@predict_router.post(
    "/predict_v2", 
    tags=tag,
    response_model=PredictionResult,
    responses=error_responses
)
def predict_route_v2(data: PredictionParams) -> PredictionResult:
    try:
        pred = Predict(params=data).predict_v2()
        return PredictionResult(**pred)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": str(e), "code": 404, "hint": "model di MLflow registry tidak ditemukan"}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(e), "code": 422, "hint": "nilai input tidak sesuai"}
        )
    except TimeoutError as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={"message": str(e), "code": 504, "hint": "upstream service timeout"}
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": str(e), "code": 503, "hint": "pipeline prediksi gagal dijalankan"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": str(e), "code": 500, "hint": "unexpected error"}
        )
