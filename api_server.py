"""
FastAPI server for DocAI OCR processing
Provides async API endpoint for document processing with Qwen field extraction
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import tempfile
from pathlib import Path
import traceback

from field_extractor import QwenFieldExtractor
from surya_ocr import pdf_to_images, process_image as load_image, perform_ocr, format_as_markdown

app = FastAPI(title="DocAI OCR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[INFO] Loading Qwen model...")
extractor = QwenFieldExtractor(use_gpu=True)
print("[SUCCESS] Qwen model loaded and ready")

@app.get("/")
def read_root():
    return {
        "message": "DocAI OCR API Server",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "qwen_loaded": extractor is not None
    }

@app.post("/process-ocr")
async def process_ocr(
    file: UploadFile = File(...),
    fields: str = Form(...)
):
    """
    Process document with OCR and field extraction
    
    Args:
        file: Uploaded document (PDF, JPG, PNG, etc.)
        fields: JSON string with field definitions
               e.g., '[{"name": "invoice_number", "type": "number", "description": "..."}]'
    
    Returns:
        JSON with extracted fields and full text
    """
    temp_file_path = None
    
    try:
        try:
            fields_list = json.loads(fields)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid fields JSON")
        
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        print(f"[INFO] Processing file: {file.filename}")
        print(f"[INFO] Temp path: {temp_file_path}")
        print(f"[INFO] Fields to extract: {len(fields_list)}")
        
        print("[STEP 1] Running Surya OCR...")
        
        if temp_file_path.lower().endswith('.pdf'):
            images = pdf_to_images(temp_file_path)
        else:
            images = load_image(temp_file_path)
        
        predictions = perform_ocr(images)
        
        filename = file.filename
        markdown = format_as_markdown(predictions, filename)
        
        full_text = ""
        for page_pred in predictions:
            for line in page_pred.text_lines:
                full_text += line.text + "\n"
        
        print(f"[SUCCESS] OCR completed. Extracted {len(full_text)} characters")
        
        print("[STEP 2] Extracting fields with Qwen AI...")
        
        if fields_list and len(fields_list) > 0:
            extracted_fields = extractor.extract_fields(full_text, fields_list)
        else:
            extracted_fields = {}
        
        print(f"[SUCCESS] Field extraction completed. Extracted {len(extracted_fields)} fields")
        
        if extracted_fields:
            non_empty = sum(1 for v in extracted_fields.values() if v and str(v).strip())
            confidence_score = (non_empty / len(extracted_fields)) * 100
        else:
            confidence_score = 100.0
        
        result = {
            "success": True,
            "full_text": full_text,
            "markdown": markdown,
            "extracted_fields": extracted_fields,
            "confidence_score": round(confidence_score, 2)
        }
        
        print("[SUCCESS] Processing completed successfully")
        return result
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "full_text": "",
            "extracted_fields": {},
            "confidence_score": 0.0
        }
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

@app.post("/categorize-document")
async def categorize_document(
    file: UploadFile = File(...),
    categories: str = Form(...)
):
    """
    Categorize document using AI based on OCR text
    
    Args:
        file: Document file (PDF, JPG, PNG)
        categories: JSON string of available category names
    
    Returns:
        JSON with detected category
    """
    temp_file_path = None
    
    try:
        categories_list = json.loads(categories)
        
        if not categories_list or len(categories_list) == 0:
            raise HTTPException(status_code=400, detail="No categories provided")
        
        content = await file.read()
        suffix = Path(file.filename).suffix
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        print(f"[INFO] Categorizing file: {file.filename}")
        print(f"[INFO] Available categories: {categories_list}")
        
        print("[STEP 1] Running OCR...")
        
        if temp_file_path.lower().endswith('.pdf'):
            images = pdf_to_images(temp_file_path)
        else:
            images = load_image(temp_file_path)
        
        predictions = perform_ocr(images)
        
        full_text = ""
        for page_pred in predictions:
            for line in page_pred.text_lines:
                full_text += line.text + "\n"
        
        print(f"[SUCCESS] OCR completed. Extracted {len(full_text)} characters")
    
        print("[STEP 2] Categorizing with AI...")
        
        detected_category = extractor.categorize_document(full_text, categories_list)
        
        markdown_text = format_as_markdown(predictions, file.filename)
        
        return {
            "success": True,
            "category": detected_category,
            "full_text": full_text,
            "markdown": markdown_text,
            "text_preview": full_text[:200] + "..." if len(full_text) > 200 else full_text
        }
        
    except Exception as e:
        print(f"[ERROR] Categorization failed: {str(e)}")
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "category": categories_list[0] if categories_list else None
        }
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass

@app.post("/extract-fields")
async def extract_fields(request: dict):
    """
    Extract fields from already OCR'd text using AI
    
    Args:
        request: JSON body with 'text' and 'fields' keys
            - text: OCR extracted text
            - fields: JSON string of field definitions
    
    Returns:
        JSON with extracted fields and confidence score
    """
    try:
        text = request.get('text', '')
        fields = request.get('fields', '[]')
        
        fields_list = json.loads(fields)
        
        if not fields_list or len(fields_list) == 0:
            raise HTTPException(status_code=400, detail="No fields provided")
        
        print(f"[INFO] Extracting {len(fields_list)} fields from text ({len(text)} chars)")
        
        extracted_fields = extractor.extract_fields(text, fields_list)
        
        total_confidence = 0.0
        for value in extracted_fields.values():
            if value and str(value).strip():
                total_confidence += 1.0
        
        confidence_score = ((total_confidence / len(fields_list)) * 100) if len(fields_list) > 0 else 0.0
        
        print(f"[SUCCESS] Extracted {len(extracted_fields)} fields with confidence {confidence_score:.2f}%")
        
        return {
            "success": True,
            "extracted_fields": extracted_fields,
            "confidence_score": confidence_score
        }
        
    except Exception as e:
        print(f"[ERROR] Field extraction failed: {str(e)}")
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "extracted_fields": {},
            "confidence_score": 0.0
        }

if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("Starting DocAI OCR API Server")
    print("="*80)
    print("Server will be available at: http://localhost:8001")
    print("API docs at: http://localhost:8001/docs")
    print("="*80)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
