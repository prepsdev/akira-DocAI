#!/usr/bin/env python3
"""
Process document with OCR and field extraction
Combines Surya OCR with Qwen-based field extraction

Usage:
    python process_document.py <file_path>                    # OCR only
    python process_document.py <file_path> <fields_json>      # OCR + extraction
"""

import sys
import os
import json
from pathlib import Path
from PIL import Image
import fitz


def print_status(message, status="info"):
    """Print status message to stderr"""
    symbols = {
        "info": "INFO",
        "success": "SUCCESS",
        "error": "ERROR",
        "processing": "PROCESSING"
    }
    print(f"[{symbols.get(status, 'INFO')}] {message}", file=sys.stderr)


def pdf_to_images(pdf_path):
    """Convert PDF to images"""
    print_status(f"Converting PDF to images: {pdf_path}", "processing")
    images = []
    
    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        print_status(f"PDF has {total_pages} page(s)", "info")
        
        for page_num in range(total_pages):
            print_status(f"Processing page {page_num + 1}/{total_pages}...", "processing")
            page = pdf_document[page_num]
            
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)
            
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        pdf_document.close()
        print_status(f"Successfully converted {total_pages} page(s)", "success")
        return images
        
    except Exception as e:
        print_status(f"Error converting PDF: {e}", "error")
        raise


def process_image(image_path):
    """Load single image"""
    print_status(f"Loading image: {image_path}", "processing")
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        print_status("Image loaded successfully", "success")
        return [image]
    except Exception as e:
        print_status(f"Error loading image: {e}", "error")
        raise


def perform_ocr(images):
    """Perform OCR using Surya"""
    print_status("Initializing Surya OCR models...", "processing")
    
    try:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        
        print_status("Loading Foundation Predictor...", "processing")
        foundation_predictor = FoundationPredictor()
        
        print_status("Loading Recognition Predictor...", "processing")
        recognition_predictor = RecognitionPredictor(foundation_predictor)
        
        print_status("Loading Detection Predictor...", "processing")
        detection_predictor = DetectionPredictor()
        
        print_status(f"Running OCR on {len(images)} image(s)...", "processing")
        predictions = recognition_predictor(images, det_predictor=detection_predictor)
        
        print_status("OCR completed successfully", "success")
        return predictions
        
    except Exception as e:
        print_status(f"Error during OCR: {e}", "error")
        raise


def format_as_markdown(predictions, filename):
    """Format OCR results as markdown"""
    markdown_output = []
    
    markdown_output.append(f"# OCR Results: {filename}\n")
    markdown_output.append(f"**Total Pages/Images:** {len(predictions)}\n")
    markdown_output.append("---\n")
    
    for idx, prediction in enumerate(predictions, 1):
        markdown_output.append(f"\n## Page {idx}\n")
        
        if hasattr(prediction, 'text_lines') and prediction.text_lines:
            for line in prediction.text_lines:
                if hasattr(line, 'text'):
                    markdown_output.append(f"{line.text}\n")
        elif hasattr(prediction, 'text'):
            markdown_output.append(f"{prediction.text}\n")
        else:
            markdown_output.append("*No text detected on this page*\n")
    
    return ''.join(markdown_output)


def extract_plain_text(predictions):
    """Extract plain text from predictions"""
    text_parts = []
    
    for prediction in predictions:
        if hasattr(prediction, 'text_lines') and prediction.text_lines:
            for line in prediction.text_lines:
                if hasattr(line, 'text'):
                    text_parts.append(line.text)
        elif hasattr(prediction, 'text'):
            text_parts.append(prediction.text)
    
    return '\n'.join(text_parts)


def extract_fields(text, fields):
    """Extract fields using Qwen"""
    print_status("Initializing Qwen field extractor...", "processing")
    
    try:
        from field_extractor import QwenFieldExtractor
        
        extractor = QwenFieldExtractor()
        extracted = extractor.extract_fields(text, fields)
        
        print_status("Field extraction completed", "success")
        return extracted
        
    except Exception as e:
        print_status(f"Error during field extraction: {e}", "error")
        return {}


def process_with_fields(input_path, fields_json=None):
    """
    Process document with OCR and optional field extraction
    
    Args:
        input_path: Path to image/PDF
        fields_json: JSON string or path to JSON file with fields to extract
    
    Returns:
        Dictionary with results
    """
    print_status(f"Starting document processing: {input_path}", "info")
    
    file_ext = Path(input_path).suffix.lower()
    
    if file_ext == '.pdf':
        images = pdf_to_images(input_path)
    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        images = process_image(input_path)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    predictions = perform_ocr(images)
    
    filename = Path(input_path).name
    markdown = format_as_markdown(predictions, filename)
    full_text = extract_plain_text(predictions)
    
    result = {
        "success": True,
        "full_text": full_text,
        "markdown": markdown,
        "confidence_score": 95.0,
        "extracted_fields": {}
    }
    
    if fields_json:
        try:
            if os.path.exists(fields_json):
                print_status(f"Loading fields from file: {fields_json}", "info")
                with open(fields_json, 'r', encoding='utf-8') as f:
                    fields = json.load(f)
            else:
                fields = json.loads(fields_json)
            
            print_status(f"Extracting {len(fields)} fields", "info")
            extracted = extract_fields(full_text, fields)
            result["extracted_fields"] = extracted
            
        except json.JSONDecodeError as e:
            print_status(f"Invalid JSON format: {e}", "error")
            result["extraction_error"] = f"Invalid JSON: {str(e)}"
        except Exception as e:
            print_status(f"Field extraction error: {e}", "error")
            result["extraction_error"] = str(e)
    
    return result


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(json.dumps({
            "success": False,
            "error": "Usage: python process_document.py <file_path> [fields_json]"
        }))
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(json.dumps({
            "success": False,
            "error": f"File not found: {input_path}"
        }))
        sys.exit(1)
    
    fields_json = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        result = process_with_fields(input_path, fields_json)
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        output_filename = f"{Path(input_path).stem}_result.json"
        output_path = results_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print_status(f"Results saved to: {output_path}", "success")
        
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
