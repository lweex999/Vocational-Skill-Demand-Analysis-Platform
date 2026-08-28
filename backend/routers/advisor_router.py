"""AI career advisor — grounded Q&A over the real recruitment data.

Talks to whatever OpenAI-compatible endpoint the user configured in 个人中心
(model + base URL + key), so no provider is baked in. Grounding comes from
function calling: the model can only obtain figures by invoking the query
tools in ai_tools.py, each of which runs a real aggregate query.
"""

import json

import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..ai_tools import TOOL_SCHEMAS, run_tool
from ..auth import get_current_username
from ..database import get_db
from ..models import User, UserAIConfig
from ..schemas import AdvisorChatRequest, AdvisorChatResponse, AdvisorCitation

router = APIRouter(
    prefix="/api/advisor",
    tags=["AI职业顾问"],
    dependencies=[Depends(get_current_username)],
)

# Bounded so a model that keeps calling tools can't loop forever on the
# user's dime; 6 is plenty for "search title -> profile -> skill salary".
# Worst case is MAX_TOOL_ROUNDS * REQUEST_TIMEOUT; keep that comfortably
# under the frontend's per-request timeout for /advisor/chat (180s) so the
# browser never gives up before the server can answer or explain itself.
MAX_TOOL_ROUNDS = 4
REQUEST_TIMEOUT = 40

SYSTEM_PROMPT = """你是「职业画像分析平台」的 AI 职业顾问，服务对象主要是准备求职的大学生。

平台有一份真实的招聘数据库（约 4.7 万条真实岗位记录），你可以通过工具查询它。

硬性要求：
1. 凡是涉及具体数字的结论——岗位数量、薪资、技能热度、城市分布、学历要求——都必须先调用工具查到真实数据再回答，绝对不能凭记忆或常识编造数字。
2. 如果工具返回 found=false 或样本不足，就如实说明"平台数据里没有/样本太少"，不要用你自己的先验知识补上一个数字冒充平台数据。
3. 引用数字时说明来源口径，例如"平台数据中 Java开发 共 1120 条岗位，平均月薪约 6271 元"。
4. 可以在真实数据之外补充学习路径、方向选择这类一般性建议，但要让用户能分清"平台数据显示的事实"和"你给的建议"。
5. 用中文回答，条理清晰，适度使用列表。面向大学生，避免堆砌术语。
6. 数据本身可能有噪声（例如个别岗位薪资字段异常），如果某个数字明显不合理，可以提醒用户这是数据噪声。"""


def _get_config(db: Session, username: str) -> UserAIConfig:
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    cfg = db.query(UserAIConfig).filter(UserAIConfig.user_id == user.id).first()
    if cfg is None or not cfg.api_base_url or not cfg.api_key or not cfg.model:
        raise HTTPException(
            status_code=400,
            detail="尚未配置 AI 模型，请先到「个人中心」填写 API 地址、模型名称和 API Key",
        )
    return cfg


def _chat_completion(cfg: UserAIConfig, messages: list[dict]) -> dict:
    """One call to the user's OpenAI-compatible /chat/completions endpoint."""
    url = cfg.api_base_url.rstrip("/") + "/chat/completions"
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg.model,
                "messages": messages,
                "tools": TOOL_SCHEMAS,
                "temperature": 0.3,
                "max_tokens": 1500,
            },
            timeout=REQUEST_TIMEOUT,
            # 不走系统代理：Windows 系统代理可能指向未运行的本地端口（如 127.0.0.1:7892），
            # 会把直连国内模型（如 DeepSeek）的请求也劫持过去导致连接失败。
            proxies={"http": None, "https": None},
        )
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="调用 AI 接口超时，请稍后重试或检查 API 地址")
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"无法连接 AI 接口：{exc}")

    if resp.status_code >= 400:
        # Surface the upstream message — usually the actionable part (bad key,
        # unknown model, no function-calling support).
        detail = resp.text[:300]
        raise HTTPException(status_code=502, detail=f"AI 接口返回 {resp.status_code}：{detail}")
    try:
        return resp.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="AI 接口返回的不是合法 JSON，请确认地址是否为 OpenAI 兼容接口")


@router.get("/status")
def advisor_status(
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    """Whether this user has a usable AI config — lets the UI show a setup
    prompt instead of failing on the first question."""
    user = db.query(User).filter(User.username == username).first()
    cfg = db.query(UserAIConfig).filter(UserAIConfig.user_id == user.id).first() if user else None
    ready = bool(cfg and cfg.api_base_url and cfg.api_key and cfg.model)
    return {"configured": ready, "model": cfg.model if ready else None}


@router.post("/chat", response_model=AdvisorChatResponse)
def advisor_chat(
    body: AdvisorChatRequest,
    username: str = Depends(get_current_username),
    db: Session = Depends(get_db),
):
    cfg = _get_config(db, username)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in body.messages:
        if m.role in ("user", "assistant"):
            messages.append({"role": m.role, "content": m.content})

    citations: list[AdvisorCitation] = []

    for _ in range(MAX_TOOL_ROUNDS):
        data = _chat_completion(cfg, messages)
        choices = data.get("choices") or []
        if not choices:
            raise HTTPException(status_code=502, detail="AI 接口未返回任何回复内容")
        msg = choices[0].get("message") or {}
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            return AdvisorChatResponse(
                reply=(msg.get("content") or "").strip() or "（模型没有返回内容）",
                citations=citations,
            )

        # Echo the assistant turn back verbatim — the protocol requires the
        # tool_calls message to precede its tool results.
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or None,
            "tool_calls": tool_calls,
        })

        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}
            result = run_tool(db, name, args)
            citations.append(AdvisorCitation(tool=name, args=args))
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": json.dumps(result, ensure_ascii=False),
            })

    raise HTTPException(
        status_code=502,
        detail=f"AI 连续调用工具超过 {MAX_TOOL_ROUNDS} 轮仍未给出结论，请换个更具体的问题再试",
    )
