#!/usr/bin/env python3
"""
Qwen-based Field Extractor for DocAI
Extracts structured fields from OCR text using local Qwen LLM
"""

import re
import json
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QwenFieldExtractor:
    """Extract structured fields from text using Qwen AI model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", use_gpu: bool = True):
        """
        Initialize Qwen field extractor
        
        Args:
            model_name: HuggingFace model name (default: Qwen2.5-1.5B-Instruct)
            use_gpu: Use GPU if available (default: True)
        """
        print(f"[INFO] Loading Qwen model: {model_name}")
        
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            device_map = "auto"
            torch_dtype = torch.float16 
            print(f"[INFO] GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.device = "cpu"
            device_map = "cpu"
            torch_dtype = torch.float32
            if use_gpu:
                print("[WARNING] GPU requested but not available, using CPU")
            else:
                print("[INFO] Using CPU")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        self.model.eval()
        print(f"[SUCCESS] Qwen model loaded successfully on {self.device.upper()}")
    
    def extract_fields(self, text: str, fields: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Extract multiple fields from text
        
        Args:
            text: OCR extracted text
            fields: List of field definitions
                   e.g., [{"name": "invoice_number", "type": "number", "description": "nomor invoice"}]
        
        Returns:
            Dictionary with extracted field values
        """
        extracted = {}
        
        print(f"[INFO] Extracting {len(fields)} fields from text")
        
        for field in fields:
            field_name = field.get('name', '')
            field_type = field.get('type', 'text')
            field_description = field.get('description', '')
            
            print(f"[PROCESSING] Extracting field: {field_name} (type: {field_type})")
            if field_description:
                print(f"[INFO] Using description: {field_description}")
            
            prompt = self._create_prompt(text, field_name, field_type, field_description)
            
            value = self._query_qwen(prompt)
            
            value = self._post_process(value, field_type, text, field_name, field_description)
            
            extracted[field_name] = value
            print(f"[SUCCESS] {field_name} = {value}")
        
        return extracted
    
    def _create_prompt(self, text: str, field_name: str, field_type: str, description: str = "") -> str:
        """Create extraction prompt for Qwen"""
        
        limited_text = text[:2000] if len(text) > 2000 else text
        
        if description and description.strip():
            question = f"What is the {field_name.replace('_', ' ')}? {description}"
        else:
            if field_type == 'text':
                question = f"What is the {field_name.replace('_', ' ')}?"
            elif field_type == 'number':
                question = f"What is the {field_name.replace('_', ' ')} value?"
            elif field_type == 'date':
                question = f"What is the {field_name.replace('_', ' ')} date?"
            elif field_type == 'email':
                question = "What is the email address?"
            elif field_type == 'phone':
                question = "What is the phone number?"
            else:
                question = f"What is the {field_name.replace('_', ' ')}?"
        
        prompt = f"""Document:
{limited_text}

Question: {question}
Answer (extract exact value from document):"""
        
        return prompt
    
    def _query_qwen(self, prompt: str) -> str:
        """Query Qwen model for extraction"""
        
        messages = [
            {"role": "system", "content": "You are a precise data extraction assistant. Extract only the requested information, nothing more."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs[0]):], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _post_process(self, value: str, field_type: str, full_text: str, field_name: str = "", description: str = "") -> str:
        """Post-process extracted value based on field type"""
        
        if value == "NOT_FOUND" or not value or len(value.strip()) == 0:
            value = self._smart_regex_fallback(field_type, field_name, description, full_text)
            if not value:
                return ""
        
        if field_type == 'number':
            cleaned = re.sub(r'[^\d.,]', '', value)
            
            dot_count = cleaned.count('.')
            comma_count = cleaned.count(',')
            
            if dot_count > 1:
                cleaned = cleaned.replace('.', '')
                cleaned = cleaned.replace(',', '.')
            elif comma_count > 1:
                cleaned = cleaned.replace(',', '')
                cleaned = cleaned.replace(',', '')
            elif dot_count == 1 and comma_count == 1:
                dot_pos = cleaned.rfind('.')
                comma_pos = cleaned.rfind(',')
                if dot_pos > comma_pos:
                    cleaned = cleaned.replace(',', '')
                else:
                    cleaned = cleaned.replace('.', '').replace(',', '.')
            elif comma_count == 1:
                cleaned = cleaned.replace(',', '.')
            
            return cleaned
        
        elif field_type == 'date':
            date_patterns = [
                r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, value, re.IGNORECASE)
                if match:
                    return match.group(0)
            
            return value.strip()
        
        elif field_type == 'email':
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            match = re.search(email_pattern, value)
            if match:
                return match.group(0)
            match = re.search(email_pattern, full_text)
            if match:
                return match.group(0)
            return ""
        
        elif field_type == 'phone':
            phone_pattern = r'[\+\(]?[0-9][0-9 .\-\(\)]{8,}[0-9]'
            match = re.search(phone_pattern, value)
            if match:
                return match.group(0).strip()
            match = re.search(phone_pattern, full_text)
            if match:
                return match.group(0).strip()
            return ""
        
        return value.strip()
    
    def _smart_regex_fallback(self, field_type: str, field_name: str, description: str, text: str) -> str:
        """Smart regex fallback that considers description context"""
            
        if field_type == 'text':
            if 'customer' in description.lower() or 'pembeli' in description.lower():
                customer_match = re.search(r'CUSTOMER.*?Company\s*:\s*([^\n]+)', text, re.IGNORECASE | re.DOTALL)
                if customer_match:
                    return customer_match.group(1).strip()
                
                customer_match = re.search(r'CUSTOMER[^\n]*\n[^\n]*\n[^\n]*:\s*([^\n]+)', text, re.IGNORECASE)
                if customer_match:
                    return customer_match.group(1).strip()
            
            if 'penjual' in description.lower() or 'seller' in description.lower() or 'a/n' in description.lower():
                an_match = re.search(r'A/N\s*:\s*([^\n]+)', text, re.IGNORECASE)
                if an_match:
                    return an_match.group(1).strip()
        
        elif field_type == 'number':
            if 'total' in description.lower() or 'invoice' in field_name.lower():
                patterns = [
                    r'Total\s*Invoice\s*:?\s*([\d.,]+)',
                    r'Grand\s*Total\s*:?\s*([\d.,]+)',
                    r'Total\s*:?\s*([\d.,]+)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        return re.sub(r'[^\d.]', '', match.group(1))
            
            if 'invoice' in field_name.lower() and 'number' in field_name.lower():
                match = re.search(r'Invoice\s*No[^\d]*(\d+)', text, re.IGNORECASE)
                if match:
                    return match.group(1)
        
        elif field_type == 'email':
            match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
            return match.group(0) if match else ""
        
        elif field_type == 'phone':
            patterns = [
                r'\+62\s*\d{2,3}\s*\d{4}\s*\d{3,4}',
                r'021\s*\d{4}\s*\d{3,4}',
                r'\d{3}[-\s]?\d{4}[-\s]?\d{3,4}'
            ]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(0)
        
        return ""
    
    def categorize_document(self, text: str, available_categories: List) -> str:
        """
        Use AI to determine document category based on OCR text
        
        Args:
            text: OCR extracted text
            available_categories: List of category dicts with 'name' and 'description' keys
        
        Returns:
            Best matching category name
        """
        print(f"[INFO] Categorizing document from {len(available_categories)} categories")
        
        limited_text = text[:1000] if len(text) > 1000 else text
        
        categories_list = ""
        category_names = []
        
        for cat in available_categories:
            cat_name = cat.get('name', cat) if isinstance(cat, dict) else cat
            cat_desc = cat.get('description', '') if isinstance(cat, dict) else ''
            category_names.append(cat_name)
            
            if cat_desc:
                categories_list += f"- {cat_name}: {cat_desc}\n"
            else:
                categories_list += f"- {cat_name}\n"
        
        prompt = f"""Analyze this document and determine which category it belongs to.

Available Categories:
{categories_list}

Document Text:
{limited_text}

Based on the content, keywords, structure, and category descriptions above, which category does this document belong to?
Return ONLY the category name, nothing else.

Category:"""
        
        messages = [
            {"role": "system", "content": "You are a document classification expert. Classify documents accurately based on their content and the provided category descriptions."},
            {"role": "user", "content": prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs[0]):], 
            skip_special_tokens=True
        ).strip()
        
        response_lower = response.lower()
        
        for category_name in category_names:
            if category_name.lower() == response_lower:
                print(f"[SUCCESS] Document categorized as: {category_name}")
                return category_name
        
        for category_name in category_names:
            if category_name.lower() in response_lower or response_lower in category_name.lower():
                print(f"[SUCCESS] Document categorized as: {category_name} (partial match)")
                return category_name
        
        print(f"[WARNING] No exact match found. Using first category: {category_names[0]}")
        return category_names[0]
    
    def extract_batch(self, text: str, fields: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Extract all fields in a single batch prompt (faster but potentially less accurate)
        
        Args:
            text: OCR extracted text
            fields: List of field definitions
        
        Returns:
            Dictionary with extracted field values
        """
        print(f"[INFO] Batch extracting {len(fields)} fields")
        
        field_list = "\n".join([
            f"{i+1}. {field['name']} ({field['type']})" 
            for i, field in enumerate(fields)
        ])
        
        limited_text = text[:1500] if len(text) > 1500 else text
        
        prompt = f"""Extract the following information from this document and return as JSON.

Fields to extract:
{field_list}

Document text:
{limited_text}

Return ONLY a JSON object with the field names as keys. If a field is not found, use empty string.
Do not add any explanations.

JSON:"""
        
        messages = [
            {"role": "system", "content": "You are a data extraction assistant. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs[0]):], 
            skip_special_tokens=True
        )
        
        try:
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group(0))
                
                for field in fields:
                    field_name = field['name']
                    field_type = field['type']
                    if field_name in extracted:
                        extracted[field_name] = self._post_process(
                            extracted[field_name], 
                            field_type, 
                            text
                        )
                
                return extracted
        except json.JSONDecodeError:
            print("[WARNING] Failed to parse batch JSON, falling back to individual extraction")
        
        return self.extract_fields(text, fields)


def main():
    """Test the field extractor"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python field_extractor.py <text_file> <fields_json>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        text = f.read()
    
    with open(sys.argv[2], 'r', encoding='utf-8') as f:
        fields = json.load(f)
    
    extractor = QwenFieldExtractor()
    result = extractor.extract_fields(text, fields)
    
    print("\n" + "="*80)
    print("EXTRACTION RESULTS:")
    print("="*80)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
