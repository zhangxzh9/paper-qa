import sys
import os
import asyncio
import json
import logging
import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Union

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from paperqa.docs import Docs
from paperqa.settings import Settings, ParsingSettings, AgentSettings
from paperqa.agents import configure_cli_logging
from paperqa.agents.search import get_directory_index
from paperqa.agents.main import agent_query

os.environ['PQA_HOME'] = '/home/ubuntu/data/paper-qa-202412'
ref_language = os.environ.get("REF_LAN", "en").strip().lower()
print(f"Using reference language: {ref_language}")

if ref_language in ["en", "english"]:
    paper_directory = "/home/ubuntu/data/paper-qa-202412/my_papers/English_ref"
elif ref_language in ["zh", "chinese"]:
    paper_directory = "/home/ubuntu/data/paper-qa-202412/my_papers/Chinese_ref"
else:
    paper_directory = "/home/ubuntu/data/paper-qa-202412/my_papers"

parsing_settings = ParsingSettings(disable_doc_valid_check=True, use_doc_details=False)
agent_settings = AgentSettings(agent_llm="gpt-4o-mini", agent_llm_config={"rate_limit": {"gpt-4o-mini": "2000000 per 1 minute"}})

initSettings = Settings(
    paper_directory=paper_directory,
    llm="gpt-4o-mini",
    llm_config={"rate_limit": {"gpt-4o-mini": "2000000 per 1 minute"}},
    summary_llm="gpt-4o-mini",
    summary_llm_config={"rate_limit": {"gpt-4o-mini": "2000000 per 1 minute"}},
    embedding="text-embedding-3-small",
    temperature=0.5,
    parsing=parsing_settings,
    agent=agent_settings,
    verbosity=1
)

logger = logging.getLogger(__name__)
configure_cli_logging(initSettings)

docs = Docs()
# Global variable to store the built index
built_index = None

# Build index at startup and reuse it
@asynccontextmanager
async def lifespan(app: FastAPI):
    global built_index
    try:
        logger.info("Building paper index at startup...")
        built_index = await get_directory_index(settings=initSettings)
        logger.info(f"Index name: {initSettings.get_index_name()}")
        logger.info(f"Index files: {await built_index.index_files}")
        logger.info("Index built successfully!")
    except Exception as e:
        logger.error(f"Error building index: {e}")
    yield

app = FastAPI(lifespan=lifespan)

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.5
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # paper_directory: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None

# New models for responses API
class InputContent(BaseModel):
    text: str
    type: str = "input_text"

class InputMessage(BaseModel):
    role: str
    content: Union[str, List[InputContent]]

class ResponseRequest(BaseModel):
    model: str
    input: Union[str, List[InputMessage]]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    service_tier: Optional[str] = "auto"
    max_output_tokens: Optional[int] = None

class OutputText(BaseModel):
    type: str = "output_text"
    text: str
    annotations: List[Any] = []

class OutputMessage(BaseModel):
    type: str = "message"
    id: str
    status: str = "completed"
    role: str = "assistant"
    content: List[OutputText]

class ResponseUsage(BaseModel):
    input_tokens: int
    input_tokens_details: Dict[str, int]
    output_tokens: int
    output_tokens_details: Dict[str, int]
    total_tokens: int

class ResponseCompletion(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    status: str = "completed"
    error: Optional[Any] = None
    incomplete_details: Optional[Any] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    model: str
    output: List[OutputMessage]
    parallel_tool_calls: bool = True
    previous_response_id: Optional[str] = None
    reasoning: Dict[str, Any] = {"effort": None, "summary": None}
    store: bool = True
    temperature: float
    text: Dict[str, Any] = {"format": {"type": "text"}}
    tool_choice: str = "auto"
    tools: List[Any] = []
    top_p: float = 1.0
    truncation: str = "disabled"
    usage: ResponseUsage
    user: Optional[str] = None
    metadata: Dict[str, Any] = {}


@app.post("/v1/responses")
async def create_response(request: Request) -> ResponseCompletion:
    try:
        logger.info("Processing new response request")
        raw_body = await request.body()
        req_data = json.loads(raw_body)
        
        # Extract user input from the message format
        user_input = ""
        input_data = req_data.get("input", "")
        
        if isinstance(input_data, list):
            # Handle message format
            for message in input_data:
                if message.get("role") == "user":
                    content = message.get("content", "")
                    if isinstance(content, str):
                        user_input = content
                    elif isinstance(content, list):
                        # Handle content array format
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "input_text":
                                user_input += item.get("text", "")
                    break
        elif isinstance(input_data, str):
            user_input = input_data
        
        logger.info(f"Extracted user input: {user_input}")
        
        # Create ResponseRequest object with defaults for missing fields
        req = ResponseRequest(
            model=req_data.get("model", "gpt-4o-mini"),
            input=user_input,
            temperature=req_data.get("temperature", 1.0),
            top_p=req_data.get("top_p", 1.0),
            stream=req_data.get("stream", False),
            service_tier=req_data.get("service_tier", "auto"),
            max_output_tokens=req_data.get("max_output_tokens",None)
        )
        
    except Exception as debug_e:
        logger.error(f"DEBUG ERROR: {debug_e}")
        # Fallback: try original parsing
        raise HTTPException(status_code=400, detail=f"Request parsing error: {debug_e}")
    
    try:
        logger.info(f"Received input: {user_input}")
        query_settings = Settings(
            paper_directory=paper_directory,
            llm="gpt-4o-mini",
            llm_config={"rate_limit": {"gpt-4o-mini": "2000000 per 1 minute"}},
            summary_llm="gpt-4o-mini",
            summary_llm_config={"rate_limit": {"gpt-4o-mini": "2000000 per 1 minute"}},
            embedding="text-embedding-3-small",
            temperature=req.temperature,
            parsing=parsing_settings,
            agent=agent_settings,
            verbosity=1,
        )

        # Query the documents using the pre-built index
        # resp = await docs.aquery(
        #     query=user_input,
        #     settings=query_settings,
        # )
        resp = await agent_query(
            query=user_input,
            settings=query_settings,
        )

        print(f"resp.answer:{resp.session.answer},resp.context: {resp.session.context},resp.references:{resp.session.references}")

        # Create simple concatenated response
        structured_response = f"Answer:\n\n {resp.session.answer}\n\nRelated Reference context:\n\n {resp.session.context}\n\n References:\n\n {resp.session.references}"

        # Create output content
        output_text = OutputText(text=structured_response)
        output_message = OutputMessage(
            id=f"msg_{uuid.uuid4().hex[:50]}",
            content=[output_text]
        )

        # Estimate token usage
        input_tokens = len(req.input.split()) * 1.3
        output_tokens = len(structured_response.split()) * 1.3
        
        usage = ResponseUsage(
            input_tokens=int(input_tokens),
            input_tokens_details={"cached_tokens": 0},
            output_tokens=int(output_tokens),
            output_tokens_details={"reasoning_tokens": 0},
            total_tokens=int(input_tokens + output_tokens)
        )

        logger.info("Response created successfully")
        return ResponseCompletion(
            id=f"resp_{uuid.uuid4().hex[:50]}",
            created_at=int(time.time()),
            model=req.model,
            output=[output_message],
            temperature=req.temperature,
            usage=usage
        )

    except Exception as e:
        logger.error(f"Error in response creation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/v1/models")
async def list_models():
    """List available models - OpenAI API compatible endpoint"""
    return {
        "object": "list",
        "data": [
            {
                "id": "PAPERQA with gpt-4o-mini",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "modd"
            }
        ]
    }

@app.get("/")
async def root():
    return {"response": "Paper-QA is running", "done": True}
