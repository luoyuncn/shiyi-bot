"""FastAPI text channel"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import json
from loguru import logger

from channels.base import BaseChannel


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str
    session_id: Optional[str] = None
    enable_tools: bool = True


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    session_id: str


class SessionResponse(BaseModel):
    """Session response model"""
    session_id: str
    created_at: str
    last_active: str
    message_count: int


class OnboardingRequest(BaseModel):
    """Onboarding request payload."""
    shiyi_identity: str
    user_identity: str
    display_name: Optional[str] = None


class PendingUpdateRequest(BaseModel):
    """Pending memory update payload."""
    status: str
    cooldown_until: Optional[str] = None


class TextAPIChannel(BaseChannel):
    """FastAPI channel for HTTP/REST interface"""

    def __init__(self, config, session_manager, agent_core):
        self.config = config
        self.session_manager = session_manager
        self.agent_core = agent_core
        self.app = FastAPI(title="Shiyi API", version="2.0.0")

        # Configure CORS
        # 配置跨域策略
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.channels.get("api", {}).get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        # 注册路由
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.post("/api/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """Non-streaming chat endpoint"""
            try:
                # Get or create session
                # 获取或创建会话
                if request.session_id:
                    context = await self.session_manager.get_session(request.session_id)
                    if not context:
                        raise HTTPException(status_code=404, detail="Session not found")
                    session_id = request.session_id
                else:
                    session = await self.session_manager.create_session({"channel": "api"})
                    session_id = session.session_id

                # Save user message
                # 保存用户消息
                await self.session_manager.save_message(
                    session_id,
                    "user",
                    request.message
                )

                # Get context and process
                # 获取上下文并处理请求
                context = await self.session_manager.get_session(session_id)
                messages = await self.session_manager.prepare_messages_for_agent(
                    context.messages
                )

                # Collect full response
                # 聚合完整回复
                full_response = ""
                async for event in self.agent_core.process_message_stream(
                    messages,
                    enable_tools=request.enable_tools
                ):
                    if event["type"] == "text":
                        full_response += event["content"]

                # Save assistant message
                # 保存助手消息
                await self.session_manager.save_message(
                    session_id,
                    "assistant",
                    full_response
                )

                return ChatResponse(
                    response=full_response,
                    session_id=session_id
                )

            except Exception as e:
                logger.error(f"Chat error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/chat/stream")
        async def chat_stream(request: ChatRequest):
            """Streaming chat endpoint (JSONL format)"""
            try:
                # Get or create session
                # 获取或创建会话
                if request.session_id:
                    context = await self.session_manager.get_session(request.session_id)
                    if not context:
                        raise HTTPException(status_code=404, detail="Session not found")
                    session_id = request.session_id
                else:
                    session = await self.session_manager.create_session({"channel": "api"})
                    session_id = session.session_id

                # Save user message
                # 保存用户消息
                await self.session_manager.save_message(
                    session_id,
                    "user",
                    request.message
                )

                # Get context
                # 获取会话上下文
                context = await self.session_manager.get_session(session_id)
                messages = await self.session_manager.prepare_messages_for_agent(
                    context.messages
                )

                async def generate():
                    """Generate streaming response"""
                    full_response = ""
                    try:
                        # Send session ID first
                        # 先发送 session_id
                        yield json.dumps({
                            "type": "session",
                            "session_id": session_id
                        }, ensure_ascii=False) + "\n"

                        # Stream events
                        # 按事件流式输出
                        async for event in self.agent_core.process_message_stream(
                            messages,
                            enable_tools=request.enable_tools
                        ):
                            if event["type"] == "text":
                                full_response += event["content"]

                            yield json.dumps(event, ensure_ascii=False) + "\n"

                        # Save assistant message
                        # 保存助手消息
                        if full_response:
                            await self.session_manager.save_message(
                                session_id,
                                "assistant",
                                full_response
                            )

                    except Exception as e:
                        logger.error(f"Stream error: {e}")
                        yield json.dumps({
                            "type": "error",
                            "error": str(e)
                        }, ensure_ascii=False) + "\n"

                return StreamingResponse(
                    generate(),
                    media_type="application/x-ndjson"
                )

            except Exception as e:
                logger.error(f"Chat stream error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/sessions", response_model=list[SessionResponse])
        async def list_sessions(limit: int = 50):
            """List all sessions"""
            try:
                sessions = await self.session_manager.list_sessions(limit=limit)
                return [
                    SessionResponse(
                        session_id=s.session_id,
                        created_at=s.created_at.isoformat(),
                        last_active=s.last_active.isoformat(),
                        message_count=len(s.messages)
                    )
                    for s in sessions
                ]
            except Exception as e:
                logger.error(f"List sessions error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/sessions", response_model=SessionResponse)
        async def create_session():
            """Create a new session"""
            try:
                session = await self.session_manager.create_session({"channel": "api"})
                return SessionResponse(
                    session_id=session.session_id,
                    created_at=session.created_at.isoformat(),
                    last_active=session.last_active.isoformat(),
                    message_count=0
                )
            except Exception as e:
                logger.error(f"Create session error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/api/sessions/{session_id}")
        async def delete_session(session_id: str):
            """Delete a session"""
            try:
                await self.session_manager.delete_session(session_id)
                return {"status": "success"}
            except Exception as e:
                logger.error(f"Delete session error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/memory/user")
        async def memory_user_state():
            """Get global user memory state."""
            try:
                return await self.session_manager.get_global_user_state()
            except Exception as e:
                logger.error(f"Get memory user state error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/memory/onboarding")
        async def memory_onboarding(request: OnboardingRequest):
            """Complete identity onboarding and persist memory docs."""
            try:
                await self.session_manager.complete_identity_onboarding(
                    shiyi_identity=request.shiyi_identity,
                    user_identity=request.user_identity,
                    display_name=request.display_name,
                )
                return await self.session_manager.get_global_user_state()
            except Exception as e:
                logger.error(f"Memory onboarding error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/memory/pending")
        async def list_memory_pending(status: str = "pending", limit: int = 20):
            """List pending memory candidates."""
            try:
                records = await self.session_manager.list_memory_pending(
                    status=status,
                    limit=limit,
                )
                return [
                    {
                        "id": item.id,
                        "candidate_fact": item.candidate_fact,
                        "confidence": item.confidence,
                        "status": item.status,
                        "source_message_id": item.source_message_id,
                        "cooldown_until": item.cooldown_until.isoformat() if item.cooldown_until else None,
                        "created_at": item.created_at.isoformat(),
                        "updated_at": item.updated_at.isoformat(),
                    }
                    for item in records
                ]
            except Exception as e:
                logger.error(f"List memory pending error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/memory/pending/{pending_id}")
        async def update_memory_pending(pending_id: str, request: PendingUpdateRequest):
            """Update pending memory status."""
            try:
                cooldown_until = None
                if request.cooldown_until:
                    cooldown_until = datetime.fromisoformat(request.cooldown_until)
                await self.session_manager.update_memory_pending_status(
                    pending_id=pending_id,
                    status=request.status,
                    cooldown_until=cooldown_until,
                )
                return {"status": "ok"}
            except Exception as e:
                logger.error(f"Update memory pending error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/memory/facts")
        async def list_memory_facts(
            scope: Optional[str] = None,
            fact_type: Optional[str] = None,
            status: str = "active",
            limit: int = 100,
        ):
            """List structured memory facts."""
            try:
                records = await self.session_manager.list_memory_facts(
                    scope=scope,
                    fact_type=fact_type,
                    status=status,
                    limit=limit,
                )
                return [
                    {
                        "id": item.id,
                        "scope": item.scope,
                        "fact_type": item.fact_type,
                        "fact_key": item.fact_key,
                        "fact_value": item.fact_value,
                        "confidence": item.confidence,
                        "status": item.status,
                        "source_message_id": item.source_message_id,
                        "created_at": item.created_at.isoformat(),
                        "updated_at": item.updated_at.isoformat(),
                    }
                    for item in records
                ]
            except Exception as e:
                logger.error(f"List memory facts error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/memory/events")
        async def list_memory_events(event_type: Optional[str] = None, limit: int = 100):
            """List memory pipeline events."""
            try:
                records = await self.session_manager.list_memory_events(
                    event_type=event_type,
                    limit=limit,
                )
                return [
                    {
                        "id": item.id,
                        "event_type": item.event_type,
                        "operation_id": item.operation_id,
                        "payload": item.payload,
                        "created_at": item.created_at.isoformat(),
                    }
                    for item in records
                ]
            except Exception as e:
                logger.error(f"List memory events error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/memory/search")
        async def search_memory(q: str, limit: int = 5, mode: str = "hybrid"):
            """Search historical memory/messages with hybrid or keyword mode."""
            try:
                if mode == "keyword":
                    return await self.session_manager.search_memory_by_keyword(q, limit=limit)
                return await self.session_manager.search_memory_hybrid(q, limit=limit)
            except Exception as e:
                logger.error(f"Search memory error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/memory/metrics")
        async def memory_metrics():
            """Get memory pipeline metrics."""
            try:
                return await self.session_manager.get_memory_metrics()
            except Exception as e:
                logger.error(f"Get memory metrics error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/memory/embedding-jobs")
        async def list_embedding_jobs(status: Optional[str] = None, limit: int = 50):
            """List async embedding jobs."""
            try:
                records = await self.session_manager.list_embedding_jobs(status=status, limit=limit)
                return [
                    {
                        "id": item.id,
                        "source_type": item.source_type,
                        "source_id": item.source_id,
                        "status": item.status,
                        "retry_count": item.retry_count,
                        "next_retry_at": item.next_retry_at.isoformat() if item.next_retry_at else None,
                        "last_error": item.last_error,
                        "created_at": item.created_at.isoformat(),
                        "updated_at": item.updated_at.isoformat(),
                    }
                    for item in records
                ]
            except Exception as e:
                logger.error(f"List embedding jobs error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/memory/embedding-jobs/run")
        async def run_embedding_jobs(max_jobs: int = 20, ignore_schedule: bool = False):
            """Run embedding queue once (manual trigger)."""
            try:
                result = await self.session_manager.run_embedding_pipeline(
                    max_jobs=max_jobs,
                    ignore_schedule=ignore_schedule,
                )
                return result
            except Exception as e:
                logger.error(f"Run embedding jobs error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {"status": "healthy"}

    async def start(self):
        """Start FastAPI server"""
        import uvicorn

        api_config = self.config.channels.get("api", {})
        host = api_config.get("host", "0.0.0.0")
        port = api_config.get("port", 8000)

        logger.info(f"🌐 FastAPI 通道启动: http://{host}:{port}")
        logger.info(f"📚 API 文档: http://{host}:{port}/docs")

        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def stop(self):
        """Stop FastAPI server"""
        logger.info("FastAPI 通道已停止")
