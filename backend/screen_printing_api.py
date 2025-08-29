"""
Screen Printing Workflow API Endpoints
=====================================

FastAPI endpoints for the professional screen printing workflow
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import os
import base64
from typing import List, Dict, Any
import tempfile
import zipfile
import io

from screen_printing_workflow import workflow

# Create router for screen printing endpoints
router = APIRouter(prefix="/screen-printing", tags=["Screen Printing Workflow"])


@router.post("/workflow")
async def run_screen_printing_workflow(
    file: UploadFile = File(...),
    colors: str = Form(...)  # JSON string of colors
):
    """
    Run the complete screen printing workflow:
    1. Background removal
    2. Edge cleanup
    3. Vectorization
    4. Color separations
    """
    try:
        # Parse colors from JSON string
        import json
        colors_list = json.loads(colors)
        
        # Save uploaded file temporarily
        temp_path = f"/tmp/screen_print_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Run the complete workflow
        results = workflow.run_complete_workflow(temp_path, colors_list)
        
        # Clean up uploaded file
        os.remove(temp_path)
        
        return {
            "success": True,
            "message": "Screen printing workflow completed successfully",
            "results": {
                "background_removed": os.path.basename(results["background_removed"]),
                "edges_cleaned": os.path.basename(results["edges_cleaned"]),
                "vectorized": os.path.basename(results["vectorized"]),
                "separations_count": len(results["separations"])
            },
            "files_available": list(results.keys())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{file_type}")
async def download_workflow_file(file_type: str):
    """
    Download specific files from the workflow:
    - background_removed: PNG with transparent background
    - edges_cleaned: PNG with hard edges
    - vectorized: SVG file
    - separations: ZIP of all color separations
    """
    try:
        if file_type == "background_removed":
            file_path = workflow.results.get("background_removed")
            if not file_path or not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Background removed file not found")
            
            return FileResponse(
                file_path, 
                media_type="image/png",
                filename="background_removed.png"
            )
        
        elif file_type == "edges_cleaned":
            file_path = workflow.results.get("edges_cleaned")
            if not file_path or not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Edges cleaned file not found")
            
            return FileResponse(
                file_path, 
                media_type="image/png",
                filename="edges_cleaned.png"
            )
        
        elif file_type == "vectorized":
            file_path = workflow.results.get("vectorized")
            if not file_path or not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Vectorized file not found")
            
            return FileResponse(
                file_path, 
                media_type="image/svg+xml",
                filename="vectorized.svg"
            )
        
        elif file_type == "separations":
            separations = workflow.results.get("separations", {})
            if not separations:
                raise HTTPException(status_code=404, detail="Color separations not found")
            
            # Create ZIP file with all separations
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for color_name, color_data in separations.items():
                    file_path = color_data["file"]
                    if os.path.exists(file_path):
                        zip_file.write(file_path, f"{color_name}.png")
            
            zip_buffer.seek(0)
            
            return FileResponse(
                io.BytesIO(zip_buffer.getvalue()),
                media_type="application/zip",
                filename="color_separations.zip"
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown file type: {file_type}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/step-by-step")
async def run_workflow_step_by_step(
    file: UploadFile = File(...),
    step: str = Form(...),  # "background", "edges", "vectorize", "separations"
    colors: str = Form("[]")  # JSON string of colors (needed for separations)
):
    """
    Run individual steps of the workflow
    """
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/step_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        results = {}
        
        if step == "background":
            # Step 1: Remove background
            bg_removed = workflow.remove_background_clean(temp_path)
            results["file"] = bg_removed
            results["step"] = "background_removed"
            
        elif step == "edges":
            # Step 2: Clean edges (requires background removed image)
            edges_cleaned = workflow.cleanup_edges(temp_path)
            results["file"] = edges_cleaned
            results["step"] = "edges_cleaned"
            
        elif step == "vectorize":
            # Step 3: Vectorize (requires cleaned edges image)
            svg_file = workflow.vectorize_to_svg(temp_path)
            results["file"] = svg_file
            results["step"] = "vectorized"
            
        elif step == "separations":
            # Step 4: Color separations (requires cleaned edges image + colors)
            import json
            colors_list = json.loads(colors)
            separations = workflow.create_color_separations(temp_path, colors_list)
            results["separations"] = separations
            results["step"] = "separations"
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown step: {step}")
        
        # Clean up uploaded file
        os.remove(temp_path)
        
        return {
            "success": True,
            "step": results["step"],
            "message": f"Step '{step}' completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_workflow_status():
    """
    Get the current status of the workflow and available files
    """
    try:
        # Check what files are available
        available_files = {}
        
        if hasattr(workflow, 'results'):
            for key, file_path in workflow.results.items():
                if file_path and os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    available_files[key] = {
                        "exists": True,
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2)
                    }
                else:
                    available_files[key] = {"exists": False}
        
        return {
            "success": True,
            "workflow_status": "ready" if available_files else "no_workflow_run",
            "available_files": available_files,
            "temp_directory": workflow.temp_dir
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_workflow_files():
    """
    Clean up all temporary workflow files
    """
    try:
        workflow.cleanup_temp_files()
        
        return {
            "success": True,
            "message": "Workflow files cleaned up successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
