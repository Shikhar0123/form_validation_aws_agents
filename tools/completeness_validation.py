from .llm_client import bedrock_converse
from typing import Dict, Any, List
import traceback
import json


with open('tools/RequireDocs.json', 'r') as f:
    dispute_docs = json.load(f)


def comp_validation(dispute_type: str, explanation_validation, document_validations):
    requirements = dispute_docs.get(dispute_type, {})
    mandatory_docs = requirements.get("mandatory_documents", [])
    optional_docs = requirements.get("optional_documents", [])
    total_mandatory = len(mandatory_docs)

    # ── Step 1: Split valid vs invalid submitted docs ─────────────────────────
    valid_submitted_types: set = set()
    pre_invalid_items: List[Dict] = []

    for doc in document_validations:
        doc_type = doc.get("document_type", "Unknown")
        quality = doc.get("document_quality", 0)
        is_valid = doc.get("is_valid", False)

        if not is_valid:
            pre_invalid_items.append(
                {
                    "document_type": doc_type,
                    "reason": f"Marked invalid. Feedback: {doc.get('feedback', 'N/A')}",
                }
            )
        elif quality < 5:
            pre_invalid_items.append(
                {
                    "document_type": doc_type,
                    "reason": f"Quality too low ({quality}/10). Minimum required is 5.",
                }
            )
        else:
            valid_submitted_types.add(doc_type)

    # ── Step 2: Match valid submissions against mandatory docs ────────────────
    matched_mandatory: List[str] = []
    pre_missing_items: List[str] = []

    for mandatory_doc in mandatory_docs:
        if mandatory_doc in valid_submitted_types:
            matched_mandatory.append(mandatory_doc)
        else:
            pre_missing_items.append(mandatory_doc)

    # ── Step 3: Compute all scores deterministically ──────────────────────────
    matched_count = len(matched_mandatory)
    pre_completeness = (
        round((matched_count / total_mandatory) * 100) if total_mandatory > 0 else 100
    )
    pre_is_complete = (
        len(pre_missing_items) == 0
        and len(pre_invalid_items) == 0
        and explanation_validation.get("is_valid", False)
    )
    pre_can_proceed = not (
        not explanation_validation.get("is_valid", False) or len(pre_missing_items) >= 2
    )

    explanation_status = (
        "Valid" if explanation_validation.get("is_valid") else "Invalid/Missing"
    )


    input_schema = {
                "type": "object",
                "properties": {
                    "is_complete": {"type": "boolean"},
                    "completeness_score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "missing_items": {"type": "array", "items": {"type": "string"}},
                    "invalid_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "document_type": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["document_type", "reason"],
                        },
                    },
                    "recommendations": {
                        "type": "string",
                        "description": (
                            "2-3 sentences in second person (you/your). "
                            "Acknowledge what was submitted correctly, then state what is still missing. "
                            "If complete, confirm and explain next steps."
                        ),
                    },
                    "can_proceed": {"type": "boolean"},
                    "summary": {
                        "type": "string",
                        "description": "Exactly 2-4 words. E.g. 'Missing key documents', 'Submission complete'.",
                    },
                },
                "required": [
                    "is_complete",
                    "completeness_score",
                    "missing_items",
                    "invalid_items",
                    "recommendations",
                    "can_proceed",
                    "summary",
                ],
            }

    tools= [
        {
            "toolSpec": {
            "name": "validate_dispute_completeness",
            "description": "Generates user-friendly recommendations and summary for a chargeback dispute completeness check.",
                "inputSchema": {
                    "json": input_schema  # must be a JSON Schema dict
                },
            }
        }
    ]

    prompt_text = f"""You are a bank chargeback dispute assistant helping a customer understand their submission status.

    **Dispute Type**: {dispute_type}

    **Explanation**: {explanation_status} — {explanation_validation.get("feedback", "Not provided")}

    **Mandatory Documents Matched** ({matched_count}/{total_mandatory}):
    {chr(10).join([f"  ✅ {doc}" for doc in matched_mandatory]) if matched_mandatory else "  (none matched)"}

    **Missing Mandatory Documents**:
    {chr(10).join([f"  ❌ {doc}" for doc in pre_missing_items]) if pre_missing_items else "  (none — all covered)"}

    **Invalid / Low-Quality Documents**:
    {chr(10).join([f"  ⚠️  {i['document_type']}: {i['reason']}" for i in pre_invalid_items]) if pre_invalid_items else "  (none)"}

    The following values are ALREADY COMPUTED — copy them EXACTLY into the tool call, do NOT modify:
    is_complete        = {str(pre_is_complete).lower()}
    completeness_score = {pre_completeness}
    missing_items      = {json.dumps(pre_missing_items)}
    invalid_items      = {json.dumps(pre_invalid_items)}
    can_proceed        = {str(pre_can_proceed).lower()}

    Your ONLY task: write `recommendations` (2-3 helpful sentences) and `summary` (2-4 words), then call the tool."""

    try:
        result = bedrock_converse(
            model_id= "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
            user_message=prompt_text,
            tools=tools,
            tool_choice={"tool": {"name":"validate_dispute_completeness"}},
            inference_config={"maxTokens": 500},
            )

        return result['tool_input']

    except Exception as e:
        print(f"[validate_completeness] ERROR: {e}\n{traceback.format_exc()}")
        return {
            "is_complete": False,
            "completeness_score": 0,
            "missing_items": ["Validation error occurred"],
            "invalid_items": [],
            "recommendations": "An error occurred during validation. Please try again.",
            "can_proceed": True,
            "summary": "Validation error",
            "dispute_type_name": dispute_type,
            "error": str(e),
        }





