from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from helper_functions import process_document_universal

load_dotenv()

app = FastAPI()

class FileProcessingRequest(BaseModel):
    file_path: str
    schema_path: str
    output_path: Optional[str] = None

@app.post("/process-file/")
async def process_file(request: FileProcessingRequest):
    """Process a document file and return structured JSON data."""
    try:
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        
        if not os.path.exists(request.schema_path):
            raise HTTPException(status_code=404, detail=f"Schema file not found: {request.schema_path}")

        result = process_document_universal(request.file_path, request.schema_path)
        
        if result["status"] != "success":
            error_msg = result.get('error', 'Unknown processing error')
            raise HTTPException(status_code=500, detail=f"Processing failed: {error_msg}")

        output_filename = None
        if request.output_path:
            os.makedirs(request.output_path, exist_ok=True)
            output_filename = os.path.join(request.output_path, "processed_document.json")
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(result["data"], f, indent=2, ensure_ascii=False)

        return JSONResponse(content={
            "status": "success",
            "message": "Document processed successfully",
            "document_type": result.get("document_type"),
            "data": result["data"],  
            "validation": result.get("validation"),
            "metadata": result.get("metadata"),
            "output_path": output_filename
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Document Processing API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)