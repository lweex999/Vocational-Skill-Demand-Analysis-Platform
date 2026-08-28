import time

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..auth import get_current_username
from ..database import get_db
from ..models import User, UserAIConfig, UserProfile
from ..ai_tools import TOOL_SCHEMAS
from ..schemas import (
    AIConfigIn, AIConfigOut, AIModelsResponse, AIProbeRequest, AITestResponse,
    UserProfileIn, UserProfileOut,
)

router = APIRouter(
    prefix="/api/account",
    tags=["个人账户"],
    dependencies=[Depends(get_current_username)],
)


def _to_out(username: str, profile: UserProfile | None) -> UserProfileOut:
    if profile is None:
        return UserProfileOut(username=username, skills=[])
    return UserProfileOut(
        username=username,
        target_title=profile.target_title,
        city=profile.city,
        education=profile.education,
        experience=profile.experience,
        salary_min=profile.salary_min,
        salary_max=profile.salary_max,
        skills=profile.skills or [],
    )


@router.get("/me", response_model=UserProfileOut)
def get_my_profile(
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first() if user else None
    return _to_out(username, profile)


@router.put("/profile", response_model=UserProfileOut)
def update_my_profile(
    body: UserProfileIn,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    profile = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
    if profile is None:
        profile = UserProfile(user_id=user.id)
        db.add(profile)

    profile.target_title = body.target_title
    profile.city = body.city
    profile.education = body.education
    profile.experience = body.experience
    profile.salary_min = body.salary_min
    profile.salary_max = body.salary_max
    profile.skills = body.skills

    db.commit()
    db.refresh(profile)
    return _to_out(username, profile)


def _ai_out(cfg: UserAIConfig | None) -> AIConfigOut:
    if cfg is None:
        return AIConfigOut()
    # The key itself is deliberately never sent back to the browser — the UI
    # only needs to know whether one is stored.
    return AIConfigOut(
        api_base_url=cfg.api_base_url,
        model=cfg.model,
        has_api_key=bool(cfg.api_key),
    )


@router.get("/ai-config", response_model=AIConfigOut)
def get_ai_config(
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    cfg = db.query(UserAIConfig).filter(UserAIConfig.user_id == user.id).first() if user else None
    return _ai_out(cfg)


@router.put("/ai-config", response_model=AIConfigOut)
def update_ai_config(
    body: AIConfigIn,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    cfg = db.query(UserAIConfig).filter(UserAIConfig.user_id == user.id).first()
    if cfg is None:
        cfg = UserAIConfig(user_id=user.id)
        db.add(cfg)

    cfg.api_base_url = body.api_base_url.strip()
    cfg.model = body.model.strip()
    # Blank/omitted key means "keep what is stored", so the user does not have
    # to re-enter their key every time they change the model name.
    if body.api_key:
        cfg.api_key = body.api_key.strip()

    db.commit()
    db.refresh(cfg)
    return _ai_out(cfg)


@router.delete("/ai-config", response_model=AIConfigOut)
def delete_ai_config(
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == username).first()
    if user is not None:
        db.query(UserAIConfig).filter(UserAIConfig.user_id == user.id).delete()
        db.commit()
    return AIConfigOut()


PROBE_TIMEOUT = 20


def _resolve_probe(db: Session, username: str, body: AIProbeRequest) -> tuple[str, str, str | None]:
    """Merge request overrides with stored config.

    Lets the UI test/list models for credentials the user just typed but
    hasn't saved, while still working when only the key is already stored
    (the form never holds the real key).
    """
    user = db.query(User).filter(User.username == username).first()
    cfg = db.query(UserAIConfig).filter(UserAIConfig.user_id == user.id).first() if user else None

    base_url = (body.api_base_url or (cfg.api_base_url if cfg else None) or "").strip()
    api_key = (body.api_key or (cfg.api_key if cfg else None) or "").strip()
    model = (body.model or (cfg.model if cfg else None) or "").strip() or None

    if not base_url:
        raise HTTPException(status_code=400, detail="请先填写 API 地址")
    if not api_key:
        raise HTTPException(status_code=400, detail="请先填写 API Key")
    return base_url, api_key, model


def _probe_request(method: str, url: str, api_key: str, payload: dict | None = None):
    try:
        return requests.request(
            method, url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=PROBE_TIMEOUT,
            # 不走系统代理：Windows 系统代理可能指向未运行的本地端口（如 127.0.0.1:7892），
            # 会把直连国内模型（如 DeepSeek）的请求也劫持过去导致连接失败。
            proxies={"http": None, "https": None},
        )
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="连接超时，请检查 API 地址是否可达")
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"无法连接：{exc}")


@router.post("/ai-config/models", response_model=AIModelsResponse)
def list_ai_models(
    body: AIProbeRequest,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    """Fetch the endpoint's /models list so the user can pick instead of typing."""
    base_url, api_key, _ = _resolve_probe(db, username, body)
    resp = _probe_request("GET", base_url.rstrip("/") + "/models", api_key)
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"获取模型列表失败（{resp.status_code}）：{resp.text[:200]}",
        )
    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="返回的不是合法 JSON，请确认地址是 OpenAI 兼容接口")

    # Standard shape is {"data":[{"id":...}]}; tolerate a bare list too.
    raw = data.get("data") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise HTTPException(status_code=502, detail="返回格式不是 OpenAI 兼容的模型列表")
    models = sorted({
        str(m.get("id")) for m in raw
        if isinstance(m, dict) and m.get("id")
    } | {str(m) for m in raw if isinstance(m, str)})
    if not models:
        raise HTTPException(status_code=502, detail="该接口没有返回任何可用模型")
    return AIModelsResponse(models=models)


@router.post("/ai-config/test", response_model=AITestResponse)
def test_ai_connection(
    body: AIProbeRequest,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    """Real end-to-end check: a tiny chat completion carrying the advisor's
    tool definitions. Passing this proves the endpoint, key, model AND the
    function-calling parameter all work — which is exactly what the advisor
    needs, unlike a bare /models ping."""
    base_url, api_key, model = _resolve_probe(db, username, body)
    if not model:
        raise HTTPException(status_code=400, detail="请先选择或填写模型名称")

    started = time.monotonic()
    resp = _probe_request(
        "POST", base_url.rstrip("/") + "/chat/completions", api_key,
        {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 8,
            "tools": TOOL_SCHEMAS,
        },
    )
    latency = int((time.monotonic() - started) * 1000)

    if resp.status_code >= 400:
        detail = resp.text[:250]
        # Retry without tools so we can tell "endpoint is broken" apart from
        # "endpoint works but doesn't support function calling" — the advisor
        # needs tools, so this distinction is worth reporting precisely.
        retry = _probe_request(
            "POST", base_url.rstrip("/") + "/chat/completions", api_key,
            {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8},
        )
        if retry.status_code < 400:
            return AITestResponse(
                ok=False, model=model, latency_ms=latency, supports_tools=False,
                message="连接成功，但该模型/接口不支持 function calling（工具调用），"
                        "AI 顾问依赖它来查询真实数据，请换一个支持工具调用的模型。",
            )
        raise HTTPException(status_code=502, detail=f"接口返回 {resp.status_code}：{detail}")

    try:
        data = resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="返回的不是合法 JSON，请确认地址是 OpenAI 兼容接口")
    if not (data.get("choices") or []):
        raise HTTPException(status_code=502, detail="接口未返回 choices，可能不是标准的 OpenAI 兼容实现")

    return AITestResponse(
        ok=True, model=data.get("model") or model, latency_ms=latency, supports_tools=True,
        message=f"连接成功，模型可用，支持工具调用（耗时 {latency} ms）",
    )
