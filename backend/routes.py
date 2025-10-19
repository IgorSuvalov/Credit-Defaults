from fastapi import HTTPException, APIRouter
from .schemas import ClientData
from . import ml_service

router = APIRouter()


@router.get("/live")
def live():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    ml_service.get_model_or_503()
    return {"status": "ready", "model_uri": ml_service.model_uri}


@router.post("/score")
def score(data: ClientData):
    if float(data.employment_length) > float(data.age):
        raise HTTPException(status_code=422, detail="employment_length cannot be greater than age")

    proba_default = ml_service.predict_proba(data)

    approved = bool(proba_default < 0.5)
    return {
        "approved": approved,
    }
