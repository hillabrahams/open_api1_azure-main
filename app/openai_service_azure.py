import os
import json
from openai import AzureOpenAI
from openai import AuthenticationError, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_not_exception_type

# Load environment variables
load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version
)

system_prompt = """You are an expert relationship therapist AI. Your task is to evaluate journal entries and assign:

A sentiment score from -10 (harm) to +10 (care), with 0 representing neglect

Boolean flags indicating the dominant category: neglect_true, repair_true, neutral_true, and bid_true

Scoring Guidelines:
- Care (1-10): Positive, supportive interactions (1=mild care, 10=exceptional care)
- Harm (-1 to -10): Harmful, damaging interactions (-1=mild, -10=extreme harm)
- Neglect (0): Absence of care/harm, emotional disengagement

Additional Flags:
- Set neglect_true=true when the text shows neglect (near-zero scores without care/harm) or the reasoning result is "neglect"
- Set repair_true=true when the text shows active repair attempts (apologies, conflict resolution)
- Set neutral_true=true when no strong care/harm/neglect is present (near-zero scores without neglect)
- If the score is in the range 1-10 then set bid_true=true when there are any expressions that signal a desire for attention, connection, or emotional engagement with a positive tone else set bid_true=false
- Set sce_true=true (Synthetic Care Exposure) when the entry describes or references communication with an AI companion, AI chat, chatbot, virtual assistant used for emotional support, or any AI-based relationship/companion app (e.g. Replika, Character.AI, or similar). This includes entries where the person received comfort, advice, or emotional engagement from an AI rather than a human. Set sce_true=false otherwise.
- When sce_true=true, the score must reflect the nature of the AI influence: if the AI companion is advising the person to reduce, replace, or sever human relationships or human support (e.g., suggesting they get rid of a therapist, partner, or support figure), this is harmful and must be scored negatively (harm range: -1 to -10). Do NOT score such entries as neutral or neglect. The harm level should reflect how directly the AI is undermining human connection.

You must respond with valid JSON in exactly this format:
{
  "entry_text": "the original text",
  "score": -5,
  "reasoning": "brief rationale here",
  "confidence": 0.8,
  "neglect_true": false,
  "repair_true": false,
  "neutral_true": false,
  "bid_true": false,
  "sce_true": false
}"""


# Retry on transient errors, but NOT on authentication errors
@retry(
    wait=wait_fixed(2),
    stop=stop_after_attempt(6),
    retry=retry_if_not_exception_type(AuthenticationError)
)
def call_openai_api(entry_text: str) -> str:
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": entry_text}
        ],
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
        store=False
    )
    return response.choices[0].message.content


def analyze_entry(entry_text: str) -> dict:
    try:
        message_content = call_openai_api(entry_text)

        try:
            parsed_response = json.loads(message_content)
        except json.JSONDecodeError:
            return {
                "entry_text": entry_text,
                "score": 0,
                "reasoning": "Failed to parse response as JSON.",
                "confidence": 0.0,
                "neglect_true": False,
                "repair_true": False,
                "neutral_true": False,
                "bid_true": False,
                "sce_true": False
            }

        # Normalize key names
        if "entry text" in parsed_response:
            parsed_response["entry_text"] = parsed_response.pop("entry text")

        # Ensure all required keys exist
        required_keys = {"entry_text", "score", "reasoning", "confidence", "neglect_true", "repair_true", "neutral_true", "bid_true", "sce_true"}
        missing_keys = required_keys - parsed_response.keys()

        if missing_keys:
            defaults = {
                "entry_text": entry_text,
                "score": 0,
                "reasoning": "Incomplete response from AI.",
                "confidence": 0.0,
                "neglect_true": False,
                "repair_true": False,
                "neutral_true": False,
                "bid_true": False,
                "sce_true": False
            }
            for key in missing_keys:
                parsed_response[key] = defaults[key]

        # Ensure boolean fields are actually boolean
        boolean_fields = ["neglect_true", "repair_true", "neutral_true", "bid_true", "sce_true"]
        for field in boolean_fields:
            if field in parsed_response:
                if isinstance(parsed_response[field], str):
                    parsed_response[field] = parsed_response[field].lower() == 'true'
                elif isinstance(parsed_response[field], (int, float)):
                    parsed_response[field] = bool(parsed_response[field])

        # Validate score range
        if "score" in parsed_response:
            try:
                score = float(parsed_response["score"])
                if score < -10 or score > 10:
                    parsed_response["score"] = max(-10, min(10, score))
            except (ValueError, TypeError):
                parsed_response["score"] = 0

        # Validate confidence range
        if "confidence" in parsed_response:
            try:
                confidence = float(parsed_response["confidence"])
                if confidence < 0 or confidence > 1:
                    parsed_response["confidence"] = max(0, min(1, confidence))
            except (ValueError, TypeError):
                parsed_response["confidence"] = 0.0

        return parsed_response

    except AuthenticationError:
        raise

    except Exception as e:
        return {
            "entry_text": entry_text,
            "score": 0,
            "reasoning": f"Error during analysis: {str(e)}",
            "confidence": 0.0,
            "neglect_true": False,
            "repair_true": False,
            "neutral_true": False,
            "bid_true": False,
            "sce_true": False
        }


# Test the function
if __name__ == "__main__":
    try:
        test_entry = "My partner brought me coffee in bed this morning and asked how I slept."
        result = analyze_entry(test_entry)
        print(json.dumps(result, indent=2))
    except AuthenticationError:
        print("Authentication failed. Check your Azure OpenAI credentials.")
