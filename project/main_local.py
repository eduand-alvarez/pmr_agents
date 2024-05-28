import os
import time
import re
import random
import json
import asyncio
from typing import List, Optional, Dict
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from xgboost import XGBClassifier

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex, Settings
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

templates = Jinja2Templates(directory="templates")

# In-memory store for audit logs and total cost
audit_logs = []
total_cost = 0

# Set the OpenAI API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Define Pydantic models for data validation
class SensorData(BaseModel):
    timestamp: float
    joint_angle: float
    joint_velocity: float
    joint_torque: float

class ChatRequest(BaseModel):
    message: str

class ScheduleRequest(BaseModel):
    cost: float

# Load the trained XGBoost model
model = XGBClassifier()
model.load_model('../ml/xgboost_model.json')

# Configure embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

def get_doc_tools(file_path: str, name: str):
    """
    Get vector query and summary query tools from a document.

    Parameters
    ----------
    file_path : str
        Path to the document file.
    name : str
        Name for the document tools.

    Returns
    -------
    tuple
        A tuple containing the vector query tool and the summary tool.
    """
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=250)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)

    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
        """
        Answer questions over a given document.

        Parameters
        ----------
        query : str
            The query string to be embedded.
        page_numbers : Optional[List[str]], optional
            Filter by set of pages. Leave as None to perform a vector search over all pages.

        Returns
        -------
        str
            The response from the query.
        """
        page_numbers = page_numbers or []
        metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(metadata_dicts, condition=FilterCondition.OR)
        )
        response = query_engine.query(query)
        return response

    vector_query_tool = FunctionTool.from_defaults(name=f"vector_tool_{name}", fn=vector_query)
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=f"Useful for summarization questions related to {name}",
    )

    return vector_query_tool, summary_tool

# Papers to use for RAG tools
papers = [
    "../data/robotic_arm_maintenance_records_detailed.pdf",
    "../data/user_manual.pdf",
]

# Build RAG tools
paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

# Initialize LLM and agent
llm = OpenAI(model="gpt-3.5-turbo")
agent_worker = FunctionCallingAgentWorker.from_tools(initial_tools, llm=llm, verbose=True)
agent = AgentRunner(agent_worker)

# Shared data generator
sensor_data_store: Dict[int, Dict] = {}

def extract_maintenance_details(response):
    """
    Extracts the total cost from the agent's response.

    Parameters
    ----------
    response : str
        The string containing the agent's output.

    Returns
    -------
    float
        The total cost extracted from the response. Returns None if the cost is not found.
    """
    if response:
        
        # Use regex to find the cost in the LLM response
        cost_match = re.search(r'\$([0-9]+\.[0-9]{2})', response)
        if cost_match:
            return float(cost_match.group(1))
        else:
            print("Total cost not found in the LLM response.")
            return None
    else:
        print("LLM response not found in the response.")
        return None

def clean_and_represent_string(input_string):
    # Extract TextNode sections using regex
    nodes = re.findall(r"TextNode\((.*?)\), score=(\d+\.\d+)\)", input_string)
    
    # Process each TextNode
    processed_nodes = []
    for node_str, score in nodes:
        node_dict = {}
        node_dict['score'] = float(score)
        
        # Extract individual fields
        id_match = re.search(r"id_='(.*?)'", node_str)
        if id_match:
            node_dict['id'] = id_match.group(1)
        
        text_match = re.search(r"text='(.*?)', start_char_idx", node_str, re.DOTALL)
        if text_match:
            node_dict['text'] = text_match.group(1).replace("\\n", "\n").strip()
        
        metadata_match = re.search(r"metadata=\{(.*?)\}, excluded_embed_metadata_keys", node_str, re.DOTALL)
        if metadata_match:
            metadata_str = metadata_match.group(1).replace("'", '"')
            node_dict['metadata'] = json.loads(f"{{{metadata_str}}}")
        
        processed_nodes.append(node_dict)
    
    return processed_nodes

async def generate_sensor_data(arm_id: int) -> None:
    """
    Generate synthetic sensor data for the robotic arm.

    Parameters
    ----------
    arm_id : int
        Identifier for the robotic arm.
    """
    while True:
        data = SensorData(
            timestamp=time.time(),
            joint_angle=random.uniform(-10, 10),
            joint_velocity=random.uniform(-10, 10),
            joint_torque=random.uniform(-1, 1)
        ).dict()
        sensor_data_store[arm_id] = data
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    for arm_id in range(1, 4):  # Assuming 3 robotic arms
        asyncio.create_task(generate_sensor_data(arm_id))

def predict_maintenance(sensor_data: Dict) -> int:
    """
    Predict if maintenance is required based on sensor data.

    Parameters
    ----------
    sensor_data : dict
        Dictionary containing sensor data.

    Returns
    -------
    int
        1 if maintenance is required, 0 otherwise.
    """
    sensor_df = pd.DataFrame([sensor_data], columns=["joint_angle", "joint_velocity", "joint_torque"])
    prediction = model.predict(sensor_df)
    return int(prediction[0])

@app.get("/stream-data/{arm_id}")
async def stream_data(arm_id: int):
    """
    Stream sensor data for a specified robotic arm.

    Parameters
    ----------
    arm_id : int
        Identifier for the robotic arm.

    Returns
    -------
    StreamingResponse
        StreamingResponse object to stream the sensor data.
    """
    async def event_generator():
        while True:
            if arm_id in sensor_data_store:
                yield f"data: {json.dumps(sensor_data_store[arm_id])}\n\n"
            await asyncio.sleep(.5)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handle chat requests and provide maintenance guidance.

    Parameters
    ----------
    request : ChatRequest
        The chat request containing the user's message.

    Returns
    -------
    JSONResponse
        JSON response containing the reply from the agent.
    """
    response = str(agent.query(request.message))
    return JSONResponse(content={"reply": response})

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    """
    Serve the main page.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.

    Returns
    -------
    FileResponse
        FileResponse object to serve the index.html file.
    """
    return FileResponse("index.html")

@app.post("/schedule")
async def schedule_endpoint(request: ScheduleRequest):
    """
    Handle schedule requests.

    Parameters
    ----------
    request : ScheduleRequest
        The schedule request containing the cost of the maintenance.

    Returns
    -------
    JSONResponse
        JSON response indicating the maintenance request submission status.
    """
    global total_cost
    total_cost += request.cost
    return JSONResponse(content={"reply": "Maintenance Request has been Submitted", "total_cost": total_cost})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time data streaming.

    Parameters
    ----------
    websocket : WebSocket
        The WebSocket connection.

    Returns
    -------
    None
    """
    arm_id = int(websocket.query_params.get("arm_id", "1"))  # Default to arm 1 if not provided
    await websocket.accept()
    try:
        while True:
            if arm_id in sensor_data_store:
                data = sensor_data_store[arm_id]
                maintenance_required = predict_maintenance(data)
                if maintenance_required:
                    system_suggestion = agent.query(
                        f"""The following machine {arm_id} has been flagged for maintenance due to the following sensor readings {data}.
                        Recommend a technician, maintenance intervention, and total cost."""
                    )
                    suggestion_string = str(system_suggestion)
                    cost = extract_maintenance_details(suggestion_string)
                    retrieval_details = str(clean_and_represent_string(str(system_suggestion.source_nodes)))
                    print('**************', retrieval_details)
                    details_id = f"details-{arm_id}-{int(time.time())}"  # Unique ID for each message
                    suggestion_string += f'<button class="schedule-btn" data-cost="{cost}" onclick="scheduleMaintenance()">Schedule</button>'
                    suggestion_string += f'<button class="details-btn" data-details="{details_id}" onclick="showDetails(\'{details_id}\')">Details</button>'
                    await websocket.send_text(json.dumps({"type": "chat", 
                                                          "message": suggestion_string,
                                                          "retrieval_details": retrieval_details,
                                                          "details_id": details_id
                                                          }))
                await websocket.send_text(json.dumps({"type": "data", 
                                                      "data": data
                                                      }))
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
