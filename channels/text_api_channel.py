"""FastAPI text channel"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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


class TextAPIChannel(BaseChannel):
    """FastAPI channel for HTTP/REST interface"""

    def __init__(self, config, session_manager, agent_core):
        self.config = config
        self.session_manager = session_manager
        self.agent_core = agent_core
        self.app = FastAPI(title="Shiyi API", version="2.0.0")

        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=config.channels.get("api", {}).get("cors_origins", ["*"]),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.post("/api/chat", response_model=ChatResponse)
        async def chat(request: ChatRequest):
            """Non-streaming chat endpoint"""
            try:
                # Get or create session
                if request.session_id:
                    context = await self.session_manager.get_session(request.session_id)
                    if not context:
                        raise HTTPException(status_code=404, detail="Session not found")
                    session_id = request.session_id
                else:
                    session = await self.session_manager.create_session({"channel": "api"})
                    session_id = session.session_id

                # Save user message
                await self.session_manager.save_message(
                    session_id,
                    "user",
                    request.message
                )

                # Get context and process
                context = await self.session_manager.get_session(session_id)
                messages = context.messages

                # Collect full response
                full_response = ""
                async for event in self.agent_core.process_message_stream(
                    messages,
                    enable_tools=request.enable_tools
                ):
                    if event["type"] == "text":
                        full_response += event["content"]

                # Save assistant message
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
                if request.session_id:
                    context = await self.session_manager.get_session(request.session_id)
                    if not context:
                        raise HTTPException(status_code=404, detail="Session not found")
                    session_id = request.session_id
                else:
                    session = await self.session_manager.create_session({"channel": "api"})
                    session_id = session.session_id

                # Save user message
                await self.session_manager.save_message(
                    session_id,
                    "user",
                    request.message
                )

                # Get context
                context = await self.session_manager.get_session(session_id)
                messages = context.messages

                async def generate():
                    """Generate streaming response"""
                    full_response = ""
                    try:
                        # Send session ID first
                        yield json.dumps({
                            "type": "session",
                            "session_id": session_id
                        }, ensure_ascii=False) + "\n"

                        # Stream events
                        async for event in self.agent_core.process_message_stream(
                            messages,
                            enable_tools=request.enable_tools
                        ):
                            if event["type"] == "text":
                                full_response += event["content"]

                            yield json.dumps(event, ensure_ascii=False) + "\n"

                        # Save assistant message
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
