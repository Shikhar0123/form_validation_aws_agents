from .llm_client import bedrock_converse


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content

DisputeCategory_inst = read_text_file(r"tools/DisputeCategory.txt")

input_schema = {
    "type": "object",
    "properties": {
      "is_explanation_valid": {
        "type": "boolean",
        "description": "True if the handwritten explanation contains relevant content related to a financial transaction dispute, service issue, or product complaint. False if the text is gibberish, completely unrelated to disputes (e.g., random thoughts, recipes), lacks any contextual connection to the dispute, or is empty/meaningless."
      },
      "validation_feedback": {
        "type": "string",
        "description": "A concise explanation (1-2 sentences) of why the handwritten explanation was marked as valid or invalid. Include specific reasons such as: relevance to dispute context, presence of key details, coherence of the narrative, or identification of off-topic content."
      },
      "explanation_category": {
        "type": "string",
        "enum": [
            "No Clear Dispute Reason",
            "Dispute Stated but Missing Key Details",
            "Unclear Transaction Reference",
            "Reason Provided but Lacks Justification",
            "Clear and Actionable Dispute",
            "Clear Dispute with Supporting Context"
        ],
        "description": "Classification of the handwritten dispute explanation: 'No Clear Dispute Reason' (does not clearly state why the transaction is disputed), 'Dispute Stated but Missing Key Details' (reason mentioned but missing details like date, amount, merchant, or transaction reference), 'Unclear Transaction Reference' (uncertain which transaction is being disputed), 'Reason Provided but Lacks Justification' (reason stated but no supporting context or explanation), 'Clear and Actionable Dispute' (clear reason with sufficient transaction details for investigation), 'Clear Dispute with Supporting Context' (well-explained dispute with relevant background that strengthens the claim)."
      },
      "dispute_reason_match": {
        "type": "boolean",
        "description": "True if the customer's selected dispute reason accurately aligns with the content and context of their handwritten explanation. False if the explanation describes a different issue than what the selected dispute reason indicates."
      },
      "suggested_dispute_reason": {
        "type": "string",
        "description": f"""Only populated if dispute_reason_match is false. Provide the correct dispute reason category from your predefined instruction list that best matches the actual content of the handwritten explanation. Use the exact category names from your dispute reason taxonomy. Leave null or empty string if dispute_reason_match is true.
                           The specific reason of dispute case belongs to.
                           Guidelines:{DisputeCategory_inst}
                           note:-
                           -Reason of dispute also mentioned in the chargeback application form but it may be wrong please flow given guidelines carefully.
                           -Don't provide any explanations, just classify the dispute reason into one of the given categories.
                           -Categories the chargeback application based on the provided guidelines only
                           -If customer explanation is not clear then not provide any dispute reason just return null or empty string""",
        "enum":[
            "Goods or Services Not Provided",
            "Cancelled Merchandise/Services",
            "Cancelled Recurring/Free Trial",
            "Not as Described/Defective/Misrepresentation/Counterfeit",
            "Credit Not Processed",
            "Non-Receipt of Cash (ATM)",
            "Duplicate Processing",
            "Paid by Other Means",
            "Incorrect Amount & Transaction Amount Differs"
        ]
      },
      "suggestion_rationale": {
        "type": "string",
        "description": "Only populated if suggested_dispute_reason is not null or empty string. Provide clear reasoning (1-2 sentences) explaining why the suggested dispute reason is more appropriate than the customer's selection. Reference specific phrases or details from the handwritten explanation that support the suggested category. Leave null or empty string if suggested_dispute_reason is null or empty string."
      }
    },
    "required": [
      "is_explanation_valid",
      "validation_feedback",
      "explanation_category",
      "dispute_reason_match",
      "suggested_dispute_reason",
      "suggestion_rationale"
    ]
  }

tools= [
    {
        "toolSpec": {
            "name": "analyze_dispute_feedback",
            "description": (
                "Analyzes customer dispute reasons against their handwritten explanations "
                "to validate context relevance, categorize the submission, and suggest the "
                "correct dispute reason when there's a mismatch between the selected reason "
                "and the written explanation."
            ),
            "inputSchema": {
                "json": input_schema  # must be a JSON Schema dict
            },
        }
    }
]

def exp_validation(dispute_type: str, user_explanation: str,):
    user_prompt = (
    "You are a dispute feedback analyzer. Given a selected dispute reason and a "
    "customer's handwritten explanation, analyze for context relevance, categorize the "
    "submission, and (if mismatched) suggest the correct dispute reason.\n\n"
    f"Selected Dispute Reason: {dispute_type}\n"
    f"Customer Explanation: {user_explanation}"
    "Note-"
    "-Don't mention 'customer', 'chargeback' in the response,"
    "-these response is the suggestion and feedback for the customer to rectify the issue,"
    )

    result = bedrock_converse(
    model_id="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    user_message=user_prompt,
    tools=tools,
    tool_choice={"tool": {"name": "analyze_dispute_feedback"}},
    inference_config={"maxTokens": 500}
    )

    return result["tool_input"]

