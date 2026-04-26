import streamlit as st
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig

st.title("Jump Counter Live Stream")

# Initialize client (Use secrets for API keys in production)
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key="YOUR_API_KEY_HERE"
)

# UI Elements
st.write("Click 'Start' to begin the stream.")
frame_placeholder = st.empty()
stats_placeholder = st.empty()

if st.button("Start"):
    # Configure video source
    source = WebcamSource(resolution=(1280, 720))
    
    config = StreamConfig(
        stream_output=["annotated_image"],
        data_output=["jump_count", "max_height_px", "updated_stats"],
        processing_timeout=3600,
        requested_plan="webrtc-gpu-medium",
        requested_region="us"
    )

    # Initialize Session
    session = client.webrtc.stream(
        source=source,
        workflow="jump-counter-and-height-analyzer-1777140092922",
        workspace="stems-workspace-rf9zm",
        image_input="image",
        config=config
    )

    # Handle incoming frames
    @session.on_frame
    def show_frame(frame, metadata):
        # Convert frame color for Streamlit (OpenCV uses BGR, PIL/Streamlit uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    # Handle prediction data
    @session.on_data()
    def on_data(data: dict, metadata):
        stats_placeholder.write(f"Stats: {data}")

    # Run the session
    session.run()
