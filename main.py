from fastapi import FastAPI

from api.routes import router as api_router

app = FastAPI(title="UrbanFixd AI Knowledge Assistant", version="0.1.0")
app.include_router(api_router)
