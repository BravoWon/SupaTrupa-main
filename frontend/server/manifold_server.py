"""
Manifold Mission Control Server
===============================

FastAPI backend that exposes the Manifold Mission Control system to the dashboard.
Handles:
- REST API for system status and control
- WebSocket for real-time data streaming
- File upload for LAS data ingestion

(c) 2026 AvgAi. All rights reserved.
"""

import sys
import os
import asyncio
import logging
import time
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pandas as pd
import json

# Add framework to path
sys.path.append("/home/ubuntu/Bestest")

from jones_framework.integration.mission_control import ManifoldMissionControl
from jones_framework.core.drilling_alpha import DrillingAlpha
from jones_framework.core.mission_replay import MissionReplay
from jones_framework.core.cognitive_cortex import CognitiveCortex
from jones_framework.core.kim import KIM
from jones_framework.core.rogue_router import RogueRouter
from jones_framework.core.geometry import GeometryEngine
from dataclasses import asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ManifoldServer")

app = FastAPI(title="Manifold Mission Control API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mission Control
# Dimension 5: Depth, ROP, WOB, RPM, Torque (example)
mission_control = ManifoldMissionControl(dimension=5)
drilling_alpha = DrillingAlpha()
mission_replay = MissionReplay()
mission_replay.start_recording("Live Session", "Well-01", ["rop", "wob", "rpm", "torque"])
kim = KIM()
geometry_engine = GeometryEngine()
rogue_router = RogueRouter(kim, drilling_alpha)
cognitive_cortex = CognitiveCortex()

# Buffer for Drilling Alpha (Depth Candles)
alpha_buffer = []
BUFFER_SIZE = 100  # Keep last 100 points for rolling window calculation

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "Manifold Mission Control Online", "status": mission_control.get_system_status()}

@app.get("/status")
async def get_status():
    return mission_control.get_system_status()

@app.get("/save_mission")
async def save_mission():
    """
    Save the current session to a .mission file.
    """
    filename = f"mission_{int(time.time())}.mission"
    filepath = f"/home/ubuntu/Bestest/data/{filename}"
    os.makedirs("/home/ubuntu/Bestest/data", exist_ok=True)
    
    size = mission_replay.save_mission(filepath)
    return {"status": "saved", "file": filename, "size_bytes": size}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive data from client (simulation mode) or just keep connection open
            # In a real scenario, this might receive control commands
            data = await websocket.receive_text()
            
            # For simulation/demo, if client sends "ping", we send status
            if data == "ping":
                await websocket.send_json({"type": "pong", "status": "alive"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def data_stream_simulation():
    """
    Simulates a real-time data stream from the rig.
    In production, this would consume from a message queue (Kafka/RabbitMQ).
    """
    logger.info("Starting Data Stream Simulation...")
    t = 0
    while True:
        # Simulate drilling data with some anomalies
        # Normal drilling
        rop = 100.0 + np.random.normal(0, 5)
        wob = 15.0 + np.random.normal(0, 1)
        
        # Inject anomaly every 50 steps
        if t % 50 == 0:
            rop += 50.0 # Spike
            
        packet = {
            "timestamp": t,
            "depth": 10000.0 + t * 0.5,
            "rop": rop,
            "wob": wob,
            "rpm": 80.0 + np.random.normal(0, 2),
            "torque": 4500.0 + np.random.normal(0, 100)
        }
        
        # Process through Mission Control
        result = await mission_control.ingest_realtime_data(packet)

        # Record to Replay System
        mission_replay.record_frame(
            timestamp=packet['timestamp'],
            depth=packet['depth'],
            data={"rop": packet['rop'], "wob": packet['wob'], "rpm": packet['rpm'], "torque": packet['torque']},
            state=result
        )

        # Process through Drilling Alpha
        alpha_buffer.append(packet)
        if len(alpha_buffer) > BUFFER_SIZE:
            alpha_buffer.pop(0)
        
        # Convert buffer to DataFrame for analysis
        df_buffer = pd.DataFrame(alpha_buffer)
        
        # 1. Compute Indicators (RSI, Bollinger) on raw data for immediate feedback
        # Rename columns to match DrillingAlpha expectations if needed, or update DrillingAlpha to be flexible
        # Assuming packet keys match: 'rop', 'depth'
        
        # Calculate RSI on ROP
        if len(df_buffer) > 15:
            rsi_series = drilling_alpha.compute_rsi(df_buffer['rop'])
            current_rsi = rsi_series.iloc[-1] if not rsi_series.empty else None
            
            bb_df = drilling_alpha.compute_bollinger_bands(df_buffer['rop'])
            current_bb = bb_df.iloc[-1].to_dict() if not bb_df.empty else None
        else:
            current_rsi = None
            current_bb = None

        # 2. Resample to Depth Candles (every 5 ft)
        # We need enough data to form a candle. 
        # In a real stream, we would accumulate until a depth step is crossed.
        # For now, let's just send the latest indicators.
        
        alpha_update = {
            "rsi": current_rsi,
            "bollinger": current_bb,
            "depth": packet['depth']
        }
        
        # 1. Auto-Adjust KIM (Learn from Surprise)
        kim.auto_adjust(result.get('surprise', 0.0), f"Depth: {packet['depth']}, ROP: {packet['rop']}")

        # 2. Route Query to Experts (MoE)
        expert_responses = rogue_router.route_query(f"Drilling at {packet['depth']}ft", result)
        
        # 3. Synthesize Final Insight via Cognitive Cortex
        # We inject the Expert Opinions into the Context
        cognitive_insight = cognitive_cortex.process_frame(result, alpha_update)
        cognitive_insight['experts'] = [asdict(e) for e in expert_responses]

        # Broadcast to Dashboard
        await manager.broadcast({
            "type": "data_update",
            "payload": result,
            "alpha": alpha_update,
            "cognitive": cognitive_insight,
            "raw": packet
        })
        
        t += 1
        await asyncio.sleep(1.0) # 1Hz update rate

@app.on_event("startup")
async def startup_event():
    # Start the simulation loop in background
    asyncio.create_task(data_stream_simulation())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
