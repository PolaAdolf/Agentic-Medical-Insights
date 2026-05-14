import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import uuid
from src.app.core.pipeline import MedicalInsightsPipeline


app = FastAPI(title="LabTest AI Agentic API")

# Initialize the pipeline globally to reuse the graph/checkpointer
medical_pipeline = MedicalInsightsPipeline()

@app.post("/analyze")
async def analyze_report(file: UploadFile = File(...)):
    # print("File Received.")
    # 1. Filter Layer: Check file format 
    allowed_extensions = ["pdf", "jpg", "jpeg", "png"]
    file_ext = file.filename.split(".")[-1].lower()
    # print("File has extension", file_ext) 
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Please upload: {allowed_extensions}"
        )

    # Read file content into bytes as required by your AgentState
    file_bytes = await file.read()
    
    # Generate a unique thread ID for this specific run
    thread_id = str(uuid.uuid4())

    # 2. Trigger the Workflow 
    final_state = await medical_pipeline.run_pipeline(
        file_bytes=file_bytes, 
        filename=file.filename,
        thread_id=thread_id
    )

    print("Pipeline final state", final_state)


    # 3. Handle Failure/Branching logic 
    if "failure" in final_state.get("steps", []) or "error" in final_state:
        return {
            "status": "failed",
            "reason": final_state.get("report_type", "Unknown error"),
            "details": "This report type is currently not supported (Only CBC supported)."
        }

    # 4. Final Output
    return {
        # "thread_id": thread_id,
        # "report_type": final_state.get("report_type"),
        # "analysis": final_state.get("final_analysis"),
        # "structured_data": final_state.get("module_2_output"),
        # "metadata": {
        #     "steps_taken": final_state.get("steps"),
        #     "ocr_status": "completed"
        # }
        "status": "Success"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)