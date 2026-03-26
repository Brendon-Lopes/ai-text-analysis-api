from fastapi import FastAPI

from app.routes.text_routes import router as text_router

app = FastAPI(title="AI Text Analysis API")


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(text_router)
