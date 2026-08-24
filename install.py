import subprocess
import sys
import os
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_python_version():
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("ERROR: Python 3.10 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    
    print("Python version is compatible")

def install_requirements():
    print_header("Installing Dependencies")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("ERROR: requirements.txt not found!")
        sys.exit(1)
    
    print("Installing packages from requirements.txt...")
    print("   This may take several minutes...\n")
    
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "--progress-bar", "on"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print("\nAll dependencies installed successfully!")
        else:
            print("\nInstallation failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError during installation: {e}")
        sys.exit(1)

def download_models():
    print_header("Downloading Chandra OCR 2 Models")
    print("Downloading models (this will take a few minutes)...")
    print("   Models are approximately 2-3 GB in size\n")
    
    try:
        print("Initializing Chandra OCR 2...")
        
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch
        
        print("  → Loading Chandra OCR 2 model...")
        model = AutoModelForImageTextToText.from_pretrained(
            "datalab-to/chandra-ocr-2",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print("  ✓ Chandra OCR 2 model loaded")
        
        print("  → Loading processor...")
        processor = AutoProcessor.from_pretrained("datalab-to/chandra-ocr-2")
        print("  ✓ Processor loaded")
        
        print("\nAll models downloaded and loaded successfully!")
        
    except Exception as e:
        print(f"\nError downloading models: {e}")
        print("\nNote: Models will be downloaded automatically on first use.")
        print("You can continue and they will download when you run the OCR script.")

def create_directories():
    print_header("Creating Directories")
    
    base_dir = Path(__file__).parent
    dirs = ['uploads', 'results']
    
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"Created/verified: {dir_name}/")

def main():
    print("\n" + "="*60)
    print("   SURYA OCR v0.17.1 - Installation Script")
    print("="*60)
    
    try:
        check_python_version()
        install_requirements()
        create_directories()
        download_models()
        
        print_header("Installation Complete!")
        print("Surya OCR is ready to use!")
        print("\nNext steps:")
        print("  1. Place images or PDFs in the uploads/ folder")
        print("  2. Run: python surya_ocr.py <filename>")
        print("  3. Or use the PHP upload interface: upload.php")
        print("\n" + "="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
