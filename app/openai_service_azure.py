import os
import json
from openai import AzureOpenAI
from openai import AuthenticationError, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_not_exception_type

# Load environment variables
load_dotenv()

# ===== DEBUG: Azure Config Loading =====
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")

print("=" * 60)
print("DEBUG: Azure OpenAI Configuration Check")
print("=" * 60)

config_ok = True

if api_key is None:
    print("❌ ERROR: AZURE_OPENAI_API_KEY not found in environment variables!")
    config_ok = False
elif api_key.strip() == "":
    print("❌ ERROR: AZURE_OPENAI_API_KEY is empty!")
    config_ok = False
else:
    print(f"✓ API Key loaded successfully")
    print(f"   Key length: {len(api_key)} characters")
    if api_key != api_key.strip():
        print("   ⚠️  WARNING: API key has leading/trailing whitespace")
        api_key = api_key.strip()
        print("   → Automatically stripped whitespace")

if azure_endpoint is None or azure_endpoint.strip() == "":
    print("❌ ERROR: AZURE_OPENAI_ENDPOINT not found in environment variables!")
    print("   Expected format: https://<your-resource-name>.openai.azure.com/")
    config_ok = False
else:
    print(f"✓ Azure Endpoint: {azure_endpoint}")

print(f"✓ API Version: {api_version}")
print(f"✓ Deployment Name: {deployment_name}")
print("=" * 60)
print()

# Initialize Azure OpenAI client
try:
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version
    )
    print("✓ Azure OpenAI client initialized successfully\n")
except Exception as e:
    print(f"❌ ERROR: Failed to initialize Azure OpenAI client: {str(e)}\n")
    raise


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

print('>>>>>>>>>Active Prompt<<<<<<<<<<\n', system_prompt, '\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

# Retry on transient errors, but NOT on authentication errors
@retry(
    wait=wait_fixed(2),
    stop=stop_after_attempt(6),
    retry=retry_if_not_exception_type(AuthenticationError)  # Don't retry auth errors
)
def call_openai_api(entry_text: str) -> str:
    try:
        print(f"\n{'='*60}")
        print("DEBUG: Making Azure OpenAI API Call")
        print(f"{'='*60}")
        print(f"Deployment: {deployment_name}")
        print(f"Entry text length: {len(entry_text)} characters")
        print(f"Temperature: 0")
        print(f"Response format: JSON")

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": entry_text}
            ],
            temperature=0,
            top_p=1,
            response_format={"type": "json_object"}
        )

        print(f"✓ API call successful!")
        print(f"Response ID: {response.id}")
        print(f"Model used: {response.model}")
        print(f"Tokens used: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")

        content = response.choices[0].message.content
        print(f"Response length: {len(content)} characters")
        print("Raw Azure OpenAI response:", content)
        print(f"{'='*60}\n")

        return content

    except AuthenticationError as e:
        print(f"\n{'='*60}")
        print("❌ AUTHENTICATION ERROR - CANNOT CONTINUE")
        print(f"{'='*60}")
        print(f"Error message: {str(e)}")
        print("\n🔑 AUTHENTICATION FAILED:")
        print("   - Your Azure API key is invalid, missing, or expired")
        print("   - Check AZURE_OPENAI_API_KEY in your App Service environment variables")
        print("   - Verify your key in the Azure Portal: portal.azure.com → Azure OpenAI → Keys and Endpoint")
        print(f"   - Current key prefix: {api_key[:10]}..." if api_key else "   - No API key loaded!")
        print(f"{'='*60}\n")
        raise

    except RateLimitError as e:
        print(f"\n{'='*60}")
        print("⏱️  RATE LIMIT ERROR")
        print(f"{'='*60}")
        print(f"Error message: {str(e)}")
        print("\n   - You're making requests too quickly")
        print("   - The function will automatically retry with backoff")
        print(f"{'='*60}\n")
        raise  # Let tenacity handle the retry

    except APIConnectionError as e:
        print(f"\n{'='*60}")
        print("🌐 CONNECTION ERROR")
        print(f"{'='*60}")
        print(f"Error message: {str(e)}")
        print("\n   - Cannot connect to Azure OpenAI endpoint")
        print(f"   - Verify AZURE_OPENAI_ENDPOINT is correct: {azure_endpoint}")
        print("   - Check Azure service health: status.azure.com")
        print(f"{'='*60}\n")
        raise  # Let tenacity handle the retry

    except APIError as e:
        print(f"\n{'='*60}")
        print("❌ AZURE OPENAI API ERROR")
        print(f"{'='*60}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")

        error_str = str(e).lower()
        if "quota" in error_str or "insufficient" in error_str:
            print("\n💳 QUOTA/BILLING ERROR:")
            print("   - You may have exceeded your Azure OpenAI quota")
            print("   - Check your quota in Azure Portal → Azure OpenAI → Quotas")
        elif "deployment" in error_str or "does not exist" in error_str:
            print("\n🤖 DEPLOYMENT ERROR:")
            print(f"   - Deployment '{deployment_name}' may not exist in your Azure OpenAI resource")
            print("   - Check available deployments in Azure Portal → Azure OpenAI → Model Deployments")

        print(f"{'='*60}\n")
        raise  # Let tenacity handle the retry

    except Exception as e:
        print(f"\n{'='*60}")
        print("❌ UNEXPECTED ERROR in API Call")
        print(f"{'='*60}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"{'='*60}\n")
        raise

def analyze_entry(entry_text: str) -> dict:
    """
    Analyzes a journal entry and returns sentiment analysis.

    IMPORTANT: Authentication errors will be raised and NOT caught.
    The calling code must handle AuthenticationError exceptions.
    """
    try:
        print(f"\n{'*'*60}")
        print(f"ANALYZING ENTRY")
        print(f"{'*'*60}")
        print(f"Entry: {entry_text[:100]}{'...' if len(entry_text) > 100 else ''}")
        print(f"{'*'*60}\n")

        # This will raise AuthenticationError if credentials are invalid
        message_content = call_openai_api(entry_text)

        try:
            parsed_response = json.loads(message_content)
            print("✓ Successfully parsed JSON response")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Azure OpenAI response as JSON: {str(e)}")
            print("Raw output:", message_content)
            return {
                "entry_text": entry_text,
                "score": 0,
                "reasoning": "Failed to parse Azure OpenAI response as JSON.",
                "confidence": 0.0,
                "neglect_true": False,
                "repair_true": False,
                "neutral_true": False,
                "bid_true": False,
                "sce_true": False
            }

        # Normalize key names (handle potential variations)
        if "entry text" in parsed_response:
            parsed_response["entry_text"] = parsed_response.pop("entry text")

        # Ensure all required keys exist
        required_keys = {"entry_text", "score", "reasoning", "confidence", "neglect_true", "repair_true", "neutral_true", "bid_true", "sce_true"}
        missing_keys = required_keys - parsed_response.keys()

        if missing_keys:
            print(f"⚠️  Missing keys in response: {missing_keys}")
            print("Response keys:", list(parsed_response.keys()))
            print("Full response:", parsed_response)

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
                    print(f"⚠️  Score out of range: {score}, clamping to [-10, 10]")
                    parsed_response["score"] = max(-10, min(10, score))
            except (ValueError, TypeError):
                print(f"❌ Invalid score format: {parsed_response['score']}")
                parsed_response["score"] = 0

        # Validate confidence range
        if "confidence" in parsed_response:
            try:
                confidence = float(parsed_response["confidence"])
                if confidence < 0 or confidence > 1:
                    print(f"⚠️  Confidence out of range: {confidence}, clamping to [0, 1]")
                    parsed_response["confidence"] = max(0, min(1, confidence))
            except (ValueError, TypeError):
                print(f"❌ Invalid confidence format: {parsed_response['confidence']}")
                parsed_response["confidence"] = 0.0

        print("✓ Analysis complete - all validations passed\n")
        return parsed_response

    except AuthenticationError:
        print("❌ Authentication error in analyze_entry - re-raising")
        raise

    except Exception as e:
        print(f"\n❌ Non-authentication exception during analyze_entry: {str(e)}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

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
    print("\n" + "="*60)
    print("STARTING TEST")
    print("="*60 + "\n")

    try:
        test_entry = "My partner brought me coffee in bed this morning and asked how I slept."
        result = analyze_entry(test_entry)

        print("\n" + "="*60)
        print("FINAL TEST RESULT")
        print("="*60)
        print(json.dumps(result, indent=2))
        print("="*60 + "\n")

    except AuthenticationError as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED DUE TO AUTHENTICATION ERROR")
        print("="*60)
        print("The program cannot continue without valid Azure OpenAI credentials.")
        print("Please fix your configuration and try again.")
        print("="*60 + "\n")
        raise
