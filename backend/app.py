from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router
app = FastAPI()

origins = ["http://localhost:5173", "http://localhost:8004", "http://127.0.0.1:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)