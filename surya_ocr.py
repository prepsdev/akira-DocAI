#!/usr/bin/env python3
import sys
import os
import json
from pathlib import Path
from PIL import Image
import fitz
import torch

def print_status(message, status="info"):
    symbols = {
        "info": "INFO",
        "success": "SUCCESS",
        "error": "ERROR",
        "processing": "PROCESSING"
    }
    print(f"[{symbols.get(status, 'INFO')}] {message}", file=sys.stderr)

def pdf_to_images(pdf_path):
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

def perform_ocr(images, use_gpu=True):
    print_status("Initializing Surya OCR models...", "processing")
    
    try:
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
            print_status(f"GPU detected: {torch.cuda.get_device_name(0)}", "info")
            print_status(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", "info")
        else:
            device = "cpu"
            if use_gpu:
                print_status("GPU requested but not available, using CPU", "info")
            else:
                print_status("Using CPU", "info")
        
        print_status("Loading Foundation Predictor...", "processing")
        foundation_predictor = FoundationPredictor(device=device)
        
        print_status("Loading Recognition Predictor...", "processing")
        recognition_predictor = RecognitionPredictor(foundation_predictor)
        
        print_status("Loading Detection Predictor...", "processing")
        detection_predictor = DetectionPredictor(device=device)
        
        print_status(f"Running OCR on {len(images)} image(s) using {device.upper()}...", "processing")
        predictions = recognition_predictor(images, det_predictor=detection_predictor)
        
        print_status("OCR completed successfully", "success")
        return predictions
        
    except Exception as e:
        print_status(f"Error during OCR: {e}", "error")
        raise

def format_as_markdown(predictions, filename):
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

def main():
    if len(sys.argv) < 2:
        print_status("Usage: python surya_ocr.py <image_or_pdf_path>", "error")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print_status(f"File not found: {input_path}", "error")
        sys.exit(1)
    
    print_status(f"Starting OCR processing for: {input_path}", "info")
    
    try:
        file_ext = Path(input_path).suffix.lower()
        
        if file_ext == '.pdf':
            images = pdf_to_images(input_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
            images = process_image(input_path)
        else:
            print_status(f"Unsupported file type: {file_ext}", "error")
            print_status("Supported formats: PDF, JPG, JPEG, PNG, BMP, TIFF, WEBP", "info")
            sys.exit(1)
        
        predictions = perform_ocr(images)
        
        filename = Path(input_path).name
        markdown_result = format_as_markdown(predictions, filename)
        
        print(markdown_result)
        
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        output_filename = f"{Path(input_path).stem}_ocr.md"
        output_path = results_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_result)
        
        print_status(f"Results saved to: {output_path}", "success")
        
    except Exception as e:
        print_status(f"OCR processing failed: {e}", "error")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
