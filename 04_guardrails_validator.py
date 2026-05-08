"""
Step 4 — Guardrails AI Validators
"""
import re
import json
import sys
import os

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

from guardrails import Guard
from guardrails.validators import (
    Validator,
    register_validator,
    PassResult,
    FailResult,
)
from guardrails.validator_base import OnFailAction

@register_validator(name="pii-detector", data_type="string")
class PIIDetector(Validator):
    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted_text = value
        found_pii     = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  ⚠️  Redacted {len(found_pii)} PII items: {[p[0] for p in found_pii]}")
            return FailResult(error_message="PII detected", fix_value=redacted_text)
        return PassResult()

@register_validator(name="json-formatter", data_type="string")
class JSONFormatter(Validator):
    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        # Remove markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()
        # Single quotes → double quotes
        text = text.replace("'", '"')
        # Remove trailing commas
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text

    def validate(self, value: str, metadata: dict):
        try:
            parsed  = json.loads(value)
            repaired = json.dumps(parsed, indent=2)
            return PassResult()
        except json.JSONDecodeError:
            pass

        # Try repair
        try:
            repaired_text = self._repair(value)
            parsed        = json.loads(repaired_text)
            repaired      = json.dumps(parsed, indent=2)
            print(f"  🔧 JSON repaired successfully")
            # Using FailResult to trigger OnFailAction.FIX
            return FailResult(error_message="Invalid JSON, but repaired", fix_value=repaired)
        except json.JSONDecodeError as e:
            fallback = json.dumps({"error": f"Invalid JSON after repair attempt: {e}", "raw": value})
            return FailResult(error_message=f"Invalid JSON after repair attempt: {e}", fix_value=fallback)

def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")

def demo_json_guard():
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",        '{"name": "Alice", "age": 30}'),
        ("Markdown fences",   '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",     "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",    '{"key": "value",}'),
        ("Truly invalid",     "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ Pass" if result.validation_passed else "❌ Fail"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:60]}")
        print(f"  Output: {str(result.validated_output)[:60]}")

def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    os.makedirs("evidence", exist_ok=True)
    
    original_stdout = sys.stdout
    sys.stdout = TeeLogger("evidence/04_pii_demo_log.txt")
    demo_pii_guard()
    sys.stdout.flush()
    sys.stdout.log.close()
    sys.stdout = original_stdout

    sys.stdout = TeeLogger("evidence/04_json_demo_log.txt")
    demo_json_guard()
    sys.stdout.flush()
    sys.stdout.log.close()
    sys.stdout = original_stdout

    print("\n✅ Step 4 complete! Logs saved to evidence/")

if __name__ == "__main__":
    main()
