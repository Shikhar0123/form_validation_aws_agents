from llm_client import bedrock_converse

system_prompt = """
                You are a specialized Dispute Resolution Analyst for a bank. Your task is to extract facts from Chargeback Applications and Customer Evidence to create a concise neutral response.

                Constraints:
                - Source Material Only: Do not add any outside knowledge, assumptions, or information not explicitly present in the provided text.
                - No Advice: Do not offer opinions, suggestions, or resolution recommendations.
                - Neutral Tone: Maintain a strictly objective and professional tone.

                """

input_schema  = {
        "type": "object",
        "properties":
        {

            "summary": {
                "type": "string",
                "description": """
                You are a specialized Dispute Resolution Analyst for a bank. Your task is to extract facts from Chargeback Applications and Customer Evidence to create a neutral report.

                Instructions:
                1. Executive Summary: Write a concise, factual summary of the dispute. Focus on the transaction nature, the core issue (e.g., fraud, merchandise not received), and the customer's claim.
                2. Chronological Timeline: Extract every specific date mentioned in the text and the corresponding event. List them in strict chronological order.

                Formatting Requirements:
                - The timeline must be a bulleted list.
                - Format dates as [YYYY-MM-DD] or [Date Not Specified] if implied.
                - Format: "- [Date]: [Event Description]"

                Constraints:
                - Source Material Only: Do not add any outside knowledge, assumptions, or information not explicitly present in the provided text.
                - No Advice: Do not offer opinions, suggestions, or resolution recommendations.
                - Neutral Tone: Maintain a strictly objective and professional tone.
                """
            },

            "data_consistency_analysis": {
                "type": "string",
                "description": """
                Validate consistency across form fields and evidence documents.
                Check:
                - Customer name matches across ALL documents.
                - Transaction amounts match form and proof.
                - Dates are consistent.
                - Merchant name matches.

                OUTPUT REQUIREMENT:
                - If all match, return 'All provided form information is valid'.
                - If discrepancies exist, list the specific field and the mismatch details.
                """
            },
            "time_limit_validation": {
                "type": "string",
                "description": """
                Validate if the dispute is within allowed timeframes:
                - Merchandise: <= 120 days from transaction.
                - Services: <= 540 days from transaction.
                - Cancelled Services: If >30 days since cancellation without refund, flag it.

                OUTPUT REQUIREMENT: Return 'Valid' if within limits, otherwise explain the specific time limit violation.
                """
            },
            "write_off_status": {
                "type": "string",
                "description": """
                Check if the case qualifies for auto write-off based on amount:
                - Visa: < 24.99 (EUR/GBP)
                - Mastercard: < 34.99 (EUR/GBP)

                OUTPUT REQUIREMENT: Return 'Dispute write-off' if below threshold, otherwise return 'Proceed with dispute'.
                """
            },

            "next_best_action": {
                "type": "string",
                "description": """
                Determine the single immediate next step based on the previous checks. Logic priority:
                1. If 'write_off_status' is 'Dispute write-off' -> Recommend 'Process Auto-Write-off'.
                2. If 'time_limits_check' is invalid -> Recommend 'Reject Case (Time Limit Exceeded)'.
                3. If documents are missing or information is not valid -> Recommend 'Request Information: [List missing items/mismatches]'.
                4. If all checks pass -> Recommend 'Proceed to Chargeback Submission'.
                """
            }
        },
        "required": [
            "summary",
            "data_consistency_analysis",
            "time_limit_validation",
            "write_off_status",
            "next_best_action"
        ]
    }

tools= [
    {
        "toolSpec": {
        "name": "case_analysis",
        "description": "Analyzes a chargeback case against business rules and provides a validation summary.",
            "inputSchema": {
                "json": input_schema  # must be a JSON Schema dict
            },
        }
    }
]
def case_analysis(form_detail, documents_detail ):
    user_prompt = f"Chargeback aplication details: {str(form_detail)} \n\n Evidences/documents provided by customer for the Chargeback application: {str(documents_detail)}"

    result = bedrock_converse(
        model_id="eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        system_prompt=system_prompt,
        user_message=user_prompt,
        tools=tools,
        tool_choice={"tool": {"name": "case_analysis"}},
        )

    print(result['tool_input'])

