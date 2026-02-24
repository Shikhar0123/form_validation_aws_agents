from tools.explantion_validation import exp_validation
from tools.document_validation import doc_validation
from tools.completeness_validation import comp_validation
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import tool
from typing import Dict, Any, List


app = BedrockAgentCoreApp()

@tool
async def validate_explanation_relevance(explanation: str, dispute_type: str, model_id: str = "global.amazon.nova-2-lite-v1:0") -> Dict[str, Any]:
    """
    AI-powered validation of explanation content for relevance and quality

    Args:
        explanation: User's written explanation
        dispute_type: Dispute type ID
        model_id: Bedrock model ID (uses same model as document validation by default)

    Returns:
        Validation result for explanation
    """
    result = exp_validation(dispute_type, explanation)

@tool
async def validate_document_with_bedrock_llms(doc_base64: str, dispute_type: str, explanation: str, media_type: str = "jpeg", model_id: str = "global.amazon.nova-2-lite-v1:0") -> Dict[str, Any]:
    """
    Validate a document using Amazon Bedrock multi-modal LLMs (Nova, Claude, etc.)

    Args:
        image_base64: Base64 encoded image
        dispute_type: Dispute type ID (1-8)
        user_explanation: User's written explanation
        image_format: Image format (jpeg, png, gif, webp)
        model_id: Bedrock model ID to use for validation

    Returns:
        Validation result with document type, validity, and feedback
    """
    result = doc_validation(dispute_type, explanation, doc_base64, media_type)

@tool
async def validate_completeness(dispute_type: str, explanation_validation: Dict[str, Any], document_validations: List[Dict[str, Any]],) -> Dict[str, Any]:
    """
    Validate completeness of all submissions for the dispute type
    Checks if user has provided all required documents and valid explanation

    Args:
        dispute_type: Dispute type ID (1-8)
        explanation_validation: Result from explanation validation
        document_validations: List of document validation results
        model_id: Bedrock model ID to use

    Returns:
        Completeness validation result with missing items and recommendations
    """
    result = comp_validation(dispute_type, explanation_validation, document_validations)



async def validate_dispute_documents(payload: Dict[str, Any]) -> Dict[str, Any]:

    action = payload.get("action", "validate_document")
    dispute_type = payload.get("dispute_type", "1")
    model_id = payload.get("model_id", "global.amazon.nova-2-lite-v1:0")

    try:
        # ----------------- DOCUMENT VALIDATION -----------------
        if action == "validate_document":
            b64 = payload.get("document_base64")
            if not b64:
                return {"success": False, "error": "No document provided"}

            img_format = payload.get("image_format", "jpeg").lower()
            img_format = "jpeg" if img_format in ["jpg", "jpeg"] else img_format
            if img_format not in ["jpeg", "png", "gif", "webp"]:
                img_format = "jpeg"

            result = await validate_document_with_bedrock_llms(
                image_base64=b64,
                dispute_type=dispute_type,
                user_explanation=payload.get("user_explanation", ""),
                image_format=img_format,
                model_id=model_id
            )

            if "error" in result:
                return {
                    "success": False,
                    "action": "validate_document",
                    "error": result["error"]
                }

            return {
                "success": True,
                "action": "validate_document",
                "validation_result": result
            }

        # ----------------- EXPLANATION VALIDATION -----------------
        if action == "validate_explanation":
            result = await validate_explanation_relevance(
                explanation=payload.get("user_explanation", ""),
                dispute_type=dispute_type,
                model_id=model_id
            )
            return {
                "success": True,
                "action": "validate_explanation",
                "validation_result": result
            }

        # ----------------- COMPLETENESS CHECK -----------------
        if action == "validate_completeness":
            result = await validate_completeness(
                dispute_type=dispute_type,
                explanation_validation=payload.get("explanation_validation", {}),
                document_validations=payload.get("document_validations", []),
                model_id=model_id
            )
            return {
                "success": True,
                "action": "validate_completeness",
                "validation_result": result
            }

        # ----------------- INVALID ACTION -----------------
        return {"success": False, "error": f"Unknown action: {action}"}

    except Exception as e:
        return {"success": False, "error": str(e)}
