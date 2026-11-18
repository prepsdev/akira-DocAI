#!/usr/bin/env python3

import os
import sys
import time
import shutil
import json
import requests
from pathlib import Path
from datetime import datetime

API_BASE_URL = "http://localhost:8000/api"
SUPPORTED_EXTENSIONS = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

def log_message(message, level="INFO"):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def get_settings():
    """Fetch directory settings from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/settings")
        if response.status_code == 200:
            settings = response.json()
            config = {}
            for setting in settings:
                if setting['setting_name'] == 'auto_upload_watch_directory':
                    config['watch_dir'] = setting.get('setting_value')
                    config['enabled'] = setting.get('is_enabled', False)
                elif setting['setting_name'] == 'auto_upload_success_directory':
                    config['success_dir'] = setting.get('setting_value')
                elif setting['setting_name'] == 'auto_upload_failed_directory':
                    config['failed_dir'] = setting.get('setting_value')
            
            return config
        else:
            log_message(f"Failed to fetch settings: {response.status_code}", "ERROR")
            return None
    except Exception as e:
        log_message(f"Error fetching settings: {str(e)}", "ERROR")
        return None

def ensure_directories(watch_dir, success_dir, failed_dir):
    """Create directories if they don't exist"""
    for directory in [watch_dir, success_dir, failed_dir]:
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            log_message(f"Directory ready: {directory}")

def upload_file(file_path):
    """Upload file to OCR API"""
    try:
        log_message(f"Uploading file: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'files[]': (os.path.basename(file_path), f)}
            
            response = requests.post(
                f"{API_BASE_URL}/ocr/upload-auto",
                files=files,
                timeout=300
            )
        
        if response.status_code in [200, 201]:
            result = response.json()
            log_message(f"Upload successful: {result.get('message', 'OK')}", "SUCCESS")
            return True, result
        else:
            log_message(f"Upload failed: {response.status_code} - {response.text}", "ERROR")
            return False, None
            
    except json.JSONDecodeError as e:
        log_message(f"Error parsing response (not JSON): {str(e)}", "ERROR")
        log_message(f"Response text: {response.text[:200]}", "ERROR")
        return False, None
    except Exception as e:
        log_message(f"Error uploading file: {str(e)}", "ERROR")
        return False, None

def move_file(source, destination_dir, status="uploaded"):
    """Move file to destination directory"""
    try:
        destination = Path(destination_dir) / os.path.basename(source)
        
        if destination.exists():
            name, ext = os.path.splitext(os.path.basename(source))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination = Path(destination_dir) / f"{name}_{timestamp}{ext}"
        
        shutil.move(source, destination)
        log_message(f"File moved to {status}: {destination}", "SUCCESS")
        return True
    except Exception as e:
        log_message(f"Error moving file: {str(e)}", "ERROR")
        return False

def scan_and_upload(watch_dir, success_dir, failed_dir):
    """Scan directory and upload all files"""
    if not os.path.exists(watch_dir):
        log_message(f"Watch directory does not exist: {watch_dir}", "ERROR")
        return 0, 0
    
    files = [f for f in Path(watch_dir).iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    if not files:
        log_message("No files found in watch directory")
        return 0, 0
    
    log_message(f"Found {len(files)} file(s) to process")
    
    success_count = 0
    failed_count = 0
    
    for file_path in files:
        log_message(f"Processing: {file_path.name}")
        
        success, result = upload_file(str(file_path))
        
        if success:
            if move_file(str(file_path), success_dir, "uploaded"):
                success_count += 1
        else:
            if move_file(str(file_path), failed_dir, "failed"):
                failed_count += 1
        
        time.sleep(1)
    
    log_message(f"Processing complete: {success_count} uploaded, {failed_count} failed", "SUCCESS")
    return success_count, failed_count

def watch_mode(watch_dir, success_dir, failed_dir, interval=30):
    """Continuously watch directory for new files"""
    log_message(f"Starting watch mode (checking every {interval} seconds)")
    log_message(f"Watch directory: {watch_dir}")
    log_message(f"Success directory: {success_dir}")
    log_message(f"Failed directory: {failed_dir}")
    log_message("Press Ctrl+C to stop")
    
    try:
        while True:
            scan_and_upload(watch_dir, success_dir, failed_dir)
            time.sleep(interval)
    except KeyboardInterrupt:
        log_message("Watch mode stopped by user", "INFO")

def main():
    """Main function"""
    log_message("=== OCR Auto-Upload System ===")
    
    config = get_settings()
    
    if not config:
        log_message("Failed to load configuration from API", "ERROR")
        sys.exit(1)
    
    if not config.get('enabled'):
        log_message("Auto-upload is disabled in settings", "INFO")
        sys.exit(0)
    
    watch_dir = config.get('watch_dir')
    success_dir = config.get('success_dir')
    failed_dir = config.get('failed_dir')
    
    if not all([watch_dir, success_dir, failed_dir]):
        log_message("Missing directory configuration", "ERROR")
        sys.exit(1)
    
    ensure_directories(watch_dir, success_dir, failed_dir)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        watch_mode(watch_dir, success_dir, failed_dir, interval)
    else:
        scan_and_upload(watch_dir, success_dir, failed_dir)

if __name__ == "__main__":
    main()
