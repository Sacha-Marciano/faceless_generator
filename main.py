from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from typing import Optional
from video_logic import generate_video

app = FastAPI()

@app.post("/generate")
def generate(
    caption: str = Form(...),
    ball1_text: str = Form(...),
    ball2_text: str = Form(...),
    duration: int = Form(10)
):
    filename = generate_video(caption, ball1_text, ball2_text, duration)
    return FileResponse(path=filename, media_type="video/mp4", filename="bouncing_balls_game.mp4")
