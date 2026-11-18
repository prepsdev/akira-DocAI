#!/usr/bin/env python3

import re
import json
import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please add it to your .env file.")
    return OpenAI(api_key=api_key)


class QwenFieldExtractor:
    """Extract structured fields from text using OpenAI API"""

    def __init__(self, model_name: str = None, use_gpu: bool = True):
        self.client = _get_client()
        self.model = os.environ.get("OPENAI_EXTRACTION_MODEL", "gpt-4o-mini")
        print(f"[INFO] Using OpenAI model for extraction: {self.model}")

    def extract_fields(self, text: str, fields: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Extract multiple fields from OCR text using OpenAI API

        Args:
            text: OCR extracted text
            fields: List of field definitions
                   e.g., [{"name": "invoice_number", "type": "number", "description": "nomor invoice"}]

        Returns:
            Dictionary with extracted field values
        """
        if not fields:
            return {}

        print(f"[INFO] Extracting {len(fields)} fields from text ({len(text)} chars)")

        # Build the field list for the prompt
        field_descriptions = []
        for f in fields:
            name = f.get('name', '')
            ftype = f.get('type', 'text')
            desc = f.get('description', '')
            if desc:
                field_descriptions.append(f'"{name}" ({ftype}): {desc}')
            else:
                field_descriptions.append(f'"{name}" ({ftype})')

        field_list = "\n".join(f"- {d}" for d in field_descriptions)
        field_names = [f.get('name', '') for f in fields]

        # Limit text to avoid token overflow
        limited_text = text[:4000] if len(text) > 4000 else text

        prompt = f"""Extract the following fields from this OCR document text.
The document may be in Indonesian or English. Field labels in the document may differ from the field names below — use the description to identify them.

Fields to extract:
{field_list}

Document text:
{limited_text}

Return a JSON object with ONLY the field names as keys and the extracted values as strings.
If a field cannot be found, use an empty string "".
Return ONLY valid JSON, no explanation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise data extraction assistant. Return only valid JSON with the requested fields."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Ensure all requested fields are present
            extracted = {}
            for name in field_names:
                value = result.get(name, "")
                extracted[name] = str(value).strip() if value else ""
                print(f"[SUCCESS] {name} = {extracted[name]}")

            return extracted

        except Exception as e:
            print(f"[ERROR] Field extraction failed: {e}")
            # Fallback: return empty values for all fields
            return {f.get('name', ''): "" for f in fields}

    def categorize_document(self, text: str, available_categories: List) -> str:
        """
        Categorize document using OpenAI based on OCR text

        Args:
            text: OCR extracted text
            available_categories: List of category dicts with 'name' and 'description' keys

        Returns:
            Best matching category name
        """
        print(f"[INFO] Categorizing document from {len(available_categories)} categories")

        limited_text = text[:1500] if len(text) > 1500 else text

        category_names = []
        categories_list = ""
        for cat in available_categories:
            cat_name = cat.get('name', cat) if isinstance(cat, dict) else cat
            cat_desc = cat.get('description', '') if isinstance(cat, dict) else ''
            category_names.append(cat_name)
            if cat_desc:
                categories_list += f"- {cat_name}: {cat_desc}\n"
            else:
                categories_list += f"- {cat_name}\n"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document classification expert. Reply with ONLY the category name, nothing else."
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Classify this document into one of these categories:\n{categories_list}\n"
                            f"Document text:\n{limited_text}\n\n"
                            "Return ONLY the exact category name."
                        )
                    }
                ],
                max_tokens=50
            )

            response_text = response.choices[0].message.content.strip()

            # Exact match first
            for cat_name in category_names:
                if cat_name.lower() == response_text.lower():
                    print(f"[SUCCESS] Document categorized as: {cat_name}")
                    return cat_name

            # Partial match
            for cat_name in category_names:
                if cat_name.lower() in response_text.lower() or response_text.lower() in cat_name.lower():
                    print(f"[SUCCESS] Document categorized as: {cat_name} (partial match)")
                    return cat_name

            print(f"[WARNING] No match found, defaulting to: {category_names[0]}")
            return category_names[0]

        except Exception as e:
            print(f"[ERROR] Categorization failed: {e}")
            return category_names[0] if category_names else ""


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
