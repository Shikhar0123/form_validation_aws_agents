from .llm_client import bedrock_converse
import json


with open('tools/RequireDocs.json', 'r') as f:
    dispute_docs = json.load(f)

system_prompt = "You are a document validation expert for a bank's chargeback processing team.\n"\
"Your role is to analyze a SINGLE submitted document and determine whether it is valid on its own merits.\n\n"\
"Key rules:\n"\
"- Evaluate ONLY the document provided. Do NOT require or expect multiple documents to be present.\n"\
"- A document is valid if it matches ANY ONE of the required documents for the dispute type and is clear, complete, and authentic.\n"\
"- Do NOT reject a document simply because other required documents have not been submitted yet. Each document is reviewed independently.\n"\
"- Reject only if: the document is unreadable/unclear, unrelated to the dispute, or shows signs of fraud (e.g., sender and recipient are the same person in an email, edited metadata, inconsistent details, etc.)\n"\
"- Be thorough, objective, and flag fraud strictly based on what is visible in this document alone.\n"


def doc_validation(dispute_type: str, user_explanation: str, document_base64: str, media_type: str = "image/jpeg"):

    required_documents = dispute_docs[dispute_type]["mandatory_documents"] + dispute_docs[dispute_type]["optional_documents"]
    document_type_enum = required_documents + ["others"]

    user_message = "Please validate the following document submitted for a chargeback dispute.\n\n"\
    f"**Dispute Type:** {dispute_type}\n\n"\
    f"**Customer Explanation:** {user_explanation}\n"

    input_schema= {
                    "type": "object",
                    "properties": {
                        "document_type": {
                            "type": "string",
                            "enum": document_type_enum,
                            "description": f"The type of document identified from the required documents list. If it doesn't match any required document, use 'others'. Possible values: {document_type_enum}"
                        },
                        "document_quality": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Quality rating of the document from 1 (lowest) to 10 (highest), based on clarity, readability, visibility of key details, and completeness."
                        },
                        "is_valid": {
                            "type": "boolean",
                            "description": "Whether the document is valid and meets the requirements for one of the required documents. Set to False if the document is unclear, unrelated, or appears fraudulent (e.g., sender and recipient are the same person in an email)."
                        },
                        "description": {
                            "type": "string",
                            "description": "A detailed description of what the document contains, including all key details such as dates, amounts, parties involved, and any other relevant information."
                        },
                        "feedback": {
                            "type": "string",
                            "description": "A clear, user-friendly message explaining the decision. If valid, confirm what document was accepted. If invalid, start with why it cannot be used, then explain what is needed instead. Tone should be neutral and professional."
                        },

                    },
                    "required": [
                        "document_type",
                        "document_quality",
                        "is_valid",
                        "description",
                        "feedback",
                    ]
                }

    tools= [
        {
            "toolSpec": {
                "name": "validate_chargeback_document",
                "description": "Validates a document submitted by a user for a bank chargeback dispute case.",
                "inputSchema": {
                    "json": input_schema  # must be a JSON Schema dict
                },
            }
        }
    ]


    result = bedrock_converse(
        model_id="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        system_prompt=system_prompt,
        user_message=user_message,
        media_base64=document_base64,
        media_type=media_type,
        tools=tools,
        tool_choice={"tool": {"name": "validate_chargeback_document"}},
        inference_config={"maxTokens": 500}
        )

    return result["tool_input"]