import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

BASE_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = (
    "あなたはコーヒー販売店の売上ダッシュボードに組み込まれたアシスタントです。"
    "日本語で簡潔に答えてください。"
)

app = FastAPI(title="Coffee Sales Dashboard API")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message が空です")

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="サーバーに OPENAI_API_KEY が設定されていません(.env を確認してください)",
        )

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            res = await client.post(OPENAI_CHAT_URL, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"OpenAI への接続に失敗しました: {exc}"
            ) from exc

    if res.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI API エラー ({res.status_code}): {res.text[:300]}",
        )

    data = res.json()
    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise HTTPException(
            status_code=502, detail="OpenAI からの応答形式が予期しないものでした"
        ) from exc

    return ChatResponse(reply=reply)


# ダッシュボードの静的ファイル(HTML/CSV)配信。API ルートの後にマウントすること。
app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")
