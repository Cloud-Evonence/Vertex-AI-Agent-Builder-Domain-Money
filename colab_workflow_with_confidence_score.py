# Step 1: Install Dependencies in Google Colab
!pip install google-generativeai pytesseract pdf2image langgraph numpy
!apt-get update
!apt-get install -y tesseract-ocr poppler-utils
!which pdfinfo
!apt-get install -y libpoppler-dev
!pip install pdfplumber
!pip install asyncio



import sys
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Callable, Any, Optional
import json
import re
import os
import asyncio
import nest_asyncio
import logging
from google.colab import files
import uuid
from datetime import datetime
from difflib import SequenceMatcher

# Apply nest_asyncio for Colab environment
nest_asyncio.apply()

# Set up logging for debugging (but use print for required output)
logging.basicConfig(filename="/tmp/document_processing.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Gemini API Configuration ---
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY", "Your API-KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    logging.info("Gemini API configured successfully.")
except Exception as e:
    logging.error(f"Error configuring Gemini API: {str(e)}")
    raise Exception(f"Gemini API configuration failed: {e}")

# --- Utility for Retries ---
async def retry_async_llm_call(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    **kwargs: Any
) -> Any:
    delay = initial_delay
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Attempt {i + 1}/{max_retries} failed: {e}")
            if i < max_retries - 1:
                print(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Max retries reached after {max_retries} attempts: {str(e)}")
                raise

# --- Document State Definition ---
class DocumentState(TypedDict):
    file_path: str
    doc_type: str
    raw_text: str
    tables: List[Dict]
    json_output: Dict
    document_id: str
    ocr_confidences: Dict
    genai_confidence_scores: Dict
    decision_status: str
    overall_confidence_score: float

# --- Output Schemas (Unchanged) ---
schemas = {
    "bank": {
        "id": "",
        "accountType": "bank_statement",
        "accountSubType": "",
        "accountHolder": "",
        "institutionName": "",
        "accountId": "",
        "accountStatement": {
            "openingBalance": None,
            "closingBalance": None,
            "startDate": "",
            "endDate": ""
        },
        "transactions": [
            {
                "transactionType": "",
                "transactionAmount": None,
                "transactionDescription": "",
                "transactionDate": "",
                "transactionMerchant": None,
                "transactionCategory": None
            }
        ],
        "accountInterestRate": None
    },
    "pay": {
        "id": "",
        "paystubNetPay": None,
        "paystubDate": "",
        "paystubGrossPay": None,
        "payPeriodstartDate": "",
        "payPeriodendDate": "",
        "paystubFrequency": "",
        "employeeName": "",
        "employer": "",
        "taxDeductions": [
            {
                "taxThisPeriod": None,
                "taxYtd": None,
                "taxType": "",
                "taxName": ""
            }
        ],
        "otherDeductions": [
            {
                "deductionThisPeriod": None,
                "deductionYtd": None,
                "deductionType": "",
                "deductionName": "",
                "deductionTaxType": ""
            }
        ]
    },
    "loan": {
        "id": "",
        "type": "",
        "accountHolder": "",
        "accountId": "",
        "institutionName": "",
        "propertyAddress": "",
        "interestRate": None,
        "principal": None,
        "interest": None,
        "escrow": None,
        "totalAmountDue": None,
        "outstandingPrincipal": None
    },
    "brokerage": {
        "id": "",
        "accounts": [
            {
                "currentValue": None,
                "accountHolder": "",
                "institutionName": "",
                "accountId": "",
                "holdings": [
                    {
                        "ticker": "",
                        "averageCost": None,
                        "holdingName": "",
                        "totalShares": None,
                        "currentValue": None,
                        "currentTickerPrice": None
                    }
                ],
                "statementDate": ""
            }
        ]
    }
}

# --- Utility Functions ---
def get_mime_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return "application/pdf"
    elif ext == ".png":
        return "image/png"
    elif ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    else:
        logging.warning(f"Unknown file extension '{ext}'. Using generic 'application/octet-stream'.")
        return "application/octet-stream"

def clean_string(value: Any) -> str:
    if value is None: return ""
    if not isinstance(value, str): value = str(value)
    cleaned = re.sub(r"[!@#$%^&*()_+=\[\]{}|;:'\",<>/?`~]", "", value)
    return re.sub(r'\s+', ' ', cleaned).strip()

def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        value = str(value)
    cleaned_value = re.sub(r"[,$€£%]", "", value).strip()
    if cleaned_value.lower() in ["n/a", "na", "--", "missing", "not available", "null", "none", ""]:
        return None
    try:
        return float(cleaned_value)
    except ValueError:
        numeric_match = re.search(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", cleaned_value)
        if numeric_match:
            try:
                return float(numeric_match.group(0).replace(',', ''))
            except ValueError:
                pass
        return None

def format_date(date_str: Any) -> str:
    if not date_str: return ""
    if isinstance(date_str, datetime):
        return date_str.strftime("%Y-%m-%d")
    if not isinstance(date_str, str): date_str = str(date_str)
    formats = [
        "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y", "%b %d %Y",
        "%Y%m%d", "%d-%m-%Y", "%d/%m/%Y", "%B %d %Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    logging.warning(f"Could not parse date: '{date_str}'")
    return ""

def fuzzy_match(a: str, b: str, threshold: float = 0.8) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def _robust_json_parse(json_string: str) -> Dict:
    json_string = json_string.strip()
    start_idx = json_string.find('{')
    end_idx = json_string.rfind('}')
    if start_idx == -1 or end_idx == -1:
        logging.warning(f"JSON string does not contain valid braces: {json_string[:100]}...")
        return {}
    json_string = json_string[start_idx : end_idx + 1]
    json_string = re.sub(r'//.*$', '', json_string, flags=re.MULTILINE)
    json_string = re.sub(r'\\\\', r'\\', json_string)
    json_string = re.sub(r'(?<=[}\]"\w])\s*(?<!\\)"\s*([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', r',"\1":', json_string)
    json_string = re.sub(r',\s*([\]}])', r'\1', json_string)
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logging.error(f"Final attempt at robust JSON parse failed for string starting: {json_string[:200]}... Error: {e}")
        return {}

# --- LangGraph Nodes ---
def fetch_document(state: DocumentState) -> DocumentState:
    print(f"Processing {state['document_id']}...")
    state["file_path"] = f"/tmp/{state['document_id']}"
    print(f"Using uploaded file at {state['file_path']}")
    logging.info(f"Using uploaded file at {state['file_path']}")
    return state

async def classify_document(state: DocumentState) -> DocumentState:
    if not state["file_path"] or not os.path.exists(state["file_path"]):
        state["doc_type"] = "bank"
        print(f"Classified as bank (fallback: no file for {state['document_id']})")
        logging.warning(f"Classified as bank (fallback: no file for {state['document_id']})")
        return state

    uploaded_file = None
    try:
        mime_type = get_mime_type(state["file_path"])
        uploaded_file = await retry_async_llm_call(asyncio.to_thread, genai.upload_file, state["file_path"], mime_type=mime_type)
        prompt = """
        Classify this document into one of the following general categories: bank, pay, loan, brokerage.
        The document could be a PDF or an image file.
        For 'loan', try to be more specific if possible: mortgage, car_loan, student_loan, other_loan.
        Your response should be a single word, chosen from the following: bank, pay, mortgage, car_loan, student_loan, other_loan, brokerage.
        If uncertain, default to 'bank'.
        """
        response = await retry_async_llm_call(asyncio.to_thread, model.generate_content, [prompt, uploaded_file])
        doc_type_raw = response.text.strip().lower()
        if "bank" in doc_type_raw:
            state["doc_type"] = "bank"
        elif "pay" in doc_type_raw:
            state["doc_type"] = "pay"
        elif "brokerage" in doc_type_raw:
            state["doc_type"] = "brokerage"
        elif "mortgage" in doc_type_raw or "car_loan" in doc_type_raw or "student_loan" in doc_type_raw or "other_loan" in doc_type_raw or "loan" in doc_type_raw:
            state["doc_type"] = "loan"
        else:
            state["doc_type"] = "bank"
            logging.warning(f"Unrecognized classification '{doc_type_raw}'. Defaulting to bank for {state['document_id']}")
        print(f"Classified as {state['doc_type']} for {state['document_id']}")
    except Exception as e:
        state["doc_type"] = "bank"
        print(f"Classification error for {state['document_id']}: {str(e)}. Defaulting to bank.")
        logging.error(f"Classification error for {state['document_id']}: {str(e)}. Defaulting to bank.")
    finally:
        if uploaded_file:
            try:
                await asyncio.to_thread(genai.delete_file, uploaded_file.name)
            except Exception as e:
                logging.warning(f"Failed to delete uploaded file {uploaded_file.name}: {e}")
    logging.info(f"Classified as {state['doc_type']} for {state['document_id']}")
    return state

async def extract_pdf(state: DocumentState) -> DocumentState:
    if not state["file_path"] or not os.path.exists(state["file_path"]):
        state["raw_text"] = ""
        print(f"Raw text extraction skipped: No file for {state['document_id']}")
        logging.warning(f"Raw text extraction skipped: No file for {state['document_id']}")
        return state

    print("Initiating OCR Extraction (Gemini-based transcription)...")
    uploaded_file = None
    try:
        mime_type = get_mime_type(state["file_path"])
        uploaded_file = await retry_async_llm_call(asyncio.to_thread, genai.upload_file, state["file_path"], mime_type=mime_type)
        prompt = """
        Provide a comprehensive and verbatim transcription of all visible text content in this document (PDF or image).
        Focus on accurately transcribing structured data like tables, lists, and clearly labeled sections.
        Crucially, if text appears to be a table, represent it in a readable, plain-text tabular format (e.g., using spaces or tabs to align columns).
        Do not interpret, summarize, or extract into JSON yet; simply provide the raw, unformatted text as it appears.
        """
        response = await retry_async_llm_call(asyncio.to_thread, model.generate_content, [prompt, uploaded_file])
        state["raw_text"] = response.text.strip()
        print(f"Gemini-based raw text extraction completed for {state['document_id']}")
        logging.info(f"Gemini-based raw text extraction completed for {state['document_id']}")
    except Exception as e:
        state["raw_text"] = ""
        print(f"Gemini-based raw text extraction failed for {state['document_id']}: {str(e)}")
        logging.error(f"Gemini-based raw text extraction failed for {state['document_id']}: {str(e)}")
    finally:
        if uploaded_file:
            try:
                await asyncio.to_thread(genai.delete_file, uploaded_file.name)
            except Exception as e:
                logging.warning(f"Failed to delete uploaded file {uploaded_file.name}: {e}")
    state["tables"] = []
    state["ocr_confidences"] = {}
    return state

async def parse_normalize(state: DocumentState) -> DocumentState:
    schema_template = schemas[state["doc_type"]].copy()
    schema_template["id"] = str(uuid.uuid4())
    prompt_parts = [
        f"Extract ALL core structured data from the provided {state['doc_type']} document (could be a PDF or an image) into a JSON object.",
        "Strictly adhere to the exact JSON schema provided below, including data types and formats. Do not deviate from the schema structure or omit required fields. If a field value is not explicitly present in the document, set it to `null` (for numbers) or an empty string `\"\"` (for strings) as per the schema.",
        "Extract every possible field from the document, including all line items for arrays (e.g., transactions, taxDeductions, holdings).",
        "Ensure all numerical values are converted to floats (e.g., '$1,234.56' to 1234.56), dates to ISO 8601 (YYYY-MM-DD, e.g., 'July 15, 2024' to '2024-07-15'), and strings are cleaned of unnecessary special characters. For multiple account holders, combine names with ' & ' (e.g., 'John Doe & Jane Smith').",
        "Output ONLY the JSON object, wrapped in ```json\n...\n```. Do NOT include any other text, explanation, or comments outside the JSON block"
    ]
    bank_categories = [
        'Utilities', 'Rent', 'Investments', 'Food', 'Shopping', 'Transfer', 'Cashback',
        'Education', 'Medical', 'Subscription/Software', 'Credit Card Payment',
        'Government/Tax', 'Investment/Brokerage', 'Pharmacy', 'Insurance', 'Other Loans',
        'Bonus', 'Interest Income', 'Payment', 'Auto Payment', 'Securities', 'Real Estate', 'Tax', 'Subscription', 'Payroll', 'Other'
    ]
    if state["doc_type"] == "bank":
        prompt_parts.extend([
            f"The 'id' field should be '{schema_template['id']}'.",
            f"Set 'accountType' to '{schema_template['accountType']}'.",
            "- Infer 'accountSubType' (checking, savings, cd) from the document. If not clear, set to `\"\"`.",
            "- Extract 'accountHolder', 'institutionName', 'accountId'. Be very careful to extract only the numerical or alphanumeric account ID itself, without any surrounding text like 'CMA' or '***' or labels like 'Account No.'.",
            "- Extract 'accountStatement' with 'openingBalance', 'closingBalance', 'startDate', 'endDate'. Set numerical fields to `null` if not found.",
            "- Extract 'transactions' array with 'transactionType' (DEBIT/CREDIT), 'transactionAmount', 'transactionDescription', 'transactionDate'. Set numerical fields to `null` if not found.",
            "- For each transaction, infer 'transactionMerchant' and 'transactionCategory' from its description. If `transactionMerchant` is not clearly stated, infer from description.",
            f"- Use these specific categories for 'transactionCategory': {', '.join(bank_categories)}. If a category doesn't fit, use 'Other'.",
            "- Extract 'accountInterestRate' (for savings/cd) if explicitly stated. Set to `null` if not found."
        ])
    elif state["doc_type"] == "pay":
        prompt_parts.extend([
            f"The 'id' field should be '{schema_template['id']}'.",
            "- Extract 'paystubNetPay', 'paystubDate', 'paystubGrossPay', 'payPeriodstartDate', 'payPeriodendDate'. Set numerical fields to `null` if not found, and dates to `\"\"` if not found.",
            "- Infer 'paystubFrequency' (monthly, semi-monthly, bi-weekly, weekly) from the document. If not clear, set to `\"\"`.",
            "- Extract 'employeeName', 'employer'.",
            "- Extract all 'taxDeductions' as an array of objects. For 'taxType', explicitly use one of 'FEDERAL', 'SOCIAL_SECURITY', 'MEDICARE', 'STATE', 'SDI', 'PFL', 'LTC', 'CITY'. If not identifiable, set to `\"\"`. For 'taxThisPeriod' and 'taxYtd', set to `null` if not present.",
            "- Extract all 'otherDeductions' as an array of objects. For 'deductionType', explicitly use one of 'PRETAX_RETIREMENT', 'MEDICAL', 'DENTAL', 'VISION', 'DEPENDENT_CARE', 'FSA', 'HSA', 'ROTH', 'ROTH_MEGA', 'ESPP', 'LIFE', 'DISABILITY', 'OTHER'. If not identifiable, set to `\"\"`. For 'deductionThisPeriod' and 'deductionYtd', set to `null` if not present. For 'deductionTaxType', use one of 'PRETAX', 'POSTTAX', 'OTHER'. If not identifiable, set to `\"\"`."
        ])
    elif state["doc_type"] == "loan":
        prompt_parts.extend([
            f"The 'id' field should be '{schema_template['id']}'.",
            "- Extract 'type' from document content: 'Mortgages', 'Car Loans', 'Student Loans', or 'Other Loans'. Prioritize explicit mentions from the document. Default to 'Other Loans' if highly uncertain.",
            "- Extract 'accountHolder', 'accountId', 'institutionName'. Be very careful to extract only the numerical or alphanumeric account ID itself, without any surrounding text like '***' or labels like 'Account No.'.",
            "- Extract 'propertyAddress' (primarily for Mortgages). Set to `\"\"` if not applicable or found.",
            "- Extract 'interestRate', 'principal' (the principal payment portion for the current period), 'interest' (current period interest payment), 'escrow' (if applicable for Mortgages). Set all numerical fields to `null` if not found.",
            "- Extract 'totalAmountDue', 'outstandingPrincipal' (the remaining balance of the loan). Set to `null` if not found."
        ])
    elif state["doc_type"] == "brokerage":
        prompt_parts.extend([
            f"The 'id' field should be '{schema_template['id']}'.",
            "**Important:** This document may contain information for multiple retirement or investment plans (e.g., 457(b), 401(k), Taxable Investing, 529). You MUST extract each distinct account as a separate object within the 'accounts' array.",
            "For each account identified, extract the following details:",
            "  - 'currentValue': The total balance for that specific account. Set to `null` if not found.",
            "  - 'accountHolder': The name of the account owner or beneficiary. This is ALWAYS a person's name (e.g., 'John A. Smith', 'Jane Doe'). It is NEVER a sentence, a URL, or instructional text. Look for labels like 'Account Owner', 'Registered To', 'Prepared For', or especially 'Beneficiary'. For 529 plans, prioritize the Beneficiary's name. If no clear name is found, leave it as an empty string `\"\"`.",
            "  - 'institutionName': The name of the financial institution.",
            "  - 'accountId': A unique identifier for the account. Extract ONLY the raw account number itself (e.g., 'A92633050-01'). Do NOT combine it with other information like the account type. Avoid extraneous characters like '***'.",
            "  - 'statementDate': The ending date of the statement period for this account. Set to `\"\"` if not found.",
            "  - 'holdings': This MUST be an array of all investment holdings within that specific account. If no holdings are listed, provide an empty array `[]`.",
            "    - For each 'holding' object:",
            "      - 'ticker': The stock ticker symbol (e.g., 'VOO', 'AAPL'). If not available, set to `\"\"`.",
            "      - 'averageCost': The average cost per share. If not available, set to `null`.",
            "      - 'holdingName': The name of the investment fund/asset (e.g., 'Vanguard S&P 500 ETF').",
            "      - 'totalShares': The number of shares/units held. Set to `null` if not found.",
            "      - 'currentValue': The current market value of this holding. Set to `null` if not found.",
            "      - 'currentTickerPrice': The current price per share/unit. If not applicable or available, set to `null`. If 'holdingName' is 'Cash Reserve' or similar, assume 'currentTickerPrice' is 1.00 and 'totalShares' is equal to 'currentValue'."
        ])
    full_prompt = "\n".join(prompt_parts) + "\n\nSchema to follow:\n" + json.dumps(schema_template, indent=2)
    extracted_data = {}
    uploaded_file = None
    try:
        mime_type = get_mime_type(state["file_path"])
        uploaded_file = await retry_async_llm_call(asyncio.to_thread, genai.upload_file, state["file_path"], mime_type=mime_type)
        response = await retry_async_llm_call(asyncio.to_thread, model.generate_content, [full_prompt, uploaded_file])
        json_string = response.text.strip()
        json_match = re.search(r"```json\n([\s\S]*?)\n```", json_string)
        if json_match:
            json_payload = json_match.group(1)
            try:
                extracted_data = json.loads(json_payload)
                extracted_data['id'] = schema_template['id']
                logging.info(f"Successfully extracted initial JSON for {state['document_id']}")
            except json.JSONDecodeError as e:
                logging.warning(f"Valid JSON block found but failed to parse for {state['document_id']}: {e}. Attempting robust salvage.")
                extracted_data = _robust_json_parse(json_payload)
                if extracted_data:
                    extracted_data['id'] = schema_template['id']
                    logging.info(f"Successfully salvaged JSON for {state['document_id']}")
                else:
                    logging.error(f"Robust JSON salvage also failed for {state['document_id']}. Original LLM output: {json_payload[:500]}...")
        else:
            logging.warning(f"No valid JSON block found in Gemini response for {state['document_id']}. Raw LLM output: {json_string[:500]}...")
    except Exception as e:
        logging.error(f"General error during core extraction for {state['document_id']}: {str(e)}")
        extracted_data = {}
    finally:
        if uploaded_file:
            try:
                await asyncio.to_thread(genai.delete_file, uploaded_file.name)
            except Exception as e:
                logging.warning(f"Failed to delete uploaded file {uploaded_file.name}: {e}")
    state["json_output"] = {
        "document_id": state["document_id"],
        "doc_type": state["doc_type"],
        "data": extracted_data
    }
    return state

async def post_process_data(state: DocumentState) -> DocumentState:
    data = state["json_output"].get("data", {})
    doc_type = state["doc_type"]
    raw_text = state["raw_text"]
    def recursive_parse_clean(item):
        if isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, (dict, list)):
                    item[k] = recursive_parse_clean(v)
                elif isinstance(v, str):
                    if any(date_key in k.lower() for date_key in ['date', 'startdate', 'enddate', 'paystubdate', 'payperiodstartdate', 'payperiodenddate', 'statementdate']):
                        item[k] = format_date(v)
                    elif any(num_key in k.lower() for num_key in ['value', 'balance', 'amount', 'pay', 'cost', 'shares', 'principal', 'interest', 'escrow', 'rate', 'ytd', 'period', 'taxthisperiod', 'deductionthisperiod', 'grosspay', 'netpay']):
                        item[k] = parse_float(v)
                    else:
                        item[k] = clean_string(v)
                elif v is None:
                    item[k] = None
            return item
        elif isinstance(item, list):
            return [recursive_parse_clean(elem) for elem in item]
        elif isinstance(item, str):
            return clean_string(item)
        return item
    state["json_output"]["data"] = recursive_parse_clean(data)
    data = state["json_output"]["data"]
    if doc_type == "brokerage":
        if "accounts" in data and isinstance(data["accounts"], list):
            unique_accounts_map = {}
            for i, account_raw in enumerate(data.get("accounts", [])):
                account = account_raw
                account_holder = account.get("accountHolder", "")
                if isinstance(account_holder, str):
                    if len(account_holder.split()) > 5 or "www." in account_holder or ".com" in account_holder or "navigate to" in account_holder:
                        logging.warning(f"Invalid 'accountHolder' detected: '{account_holder}'. Resetting and attempting to find a better one.")
                        account["accountHolder"] = ""
                        account_holder = ""
                owner_beneficiary_regex = r"(?:Beneficiary|Account Owner|Prepared For|Registered To|For the benefit of|for)\s*:\s*([A-Za-z.\s'-]+(?:[A-Za-z.\s'-]+){1,3})"
                found_name_match = re.search(owner_beneficiary_regex, raw_text, re.IGNORECASE)
                if found_name_match:
                    extracted_name = clean_string(found_name_match.group(1)).strip()
                    if len(extracted_name.split()) > 1 and not fuzzy_match(extracted_name, account_holder):
                        account["accountHolder"] = extracted_name
                        logging.info(f"Account holder updated via regex to: '{extracted_name}'")
                account_id = clean_string(account.get("accountId", ""))
                institution_name = clean_string(account.get("institutionName", ""))
                if account_id and len(account_id) > 5 and re.search(r'\d', account_id):
                    account_key_base = re.sub(r'[^a-zA-Z0-9]', '', account_id).lower()
                elif institution_name:
                    inferred_plan_type = ""
                    account_key_base = f"{institution_name.lower().replace(' ', '_')}-{inferred_plan_type}"
                else:
                    account_key_base = f"{os.path.splitext(state['document_id'])[0].split('/')[-1]}-{i}"
                current_account_holdings = []
                processed_holding_keys = set()
                for holding in account.get("holdings", []):
                    holding_name = clean_string(holding.get("holdingName", ""))
                    ticker = clean_string(holding.get("ticker", "")) if holding.get("ticker") is not None else ""
                    if holding.get("currentTickerPrice") is None and holding.get("currentValue") is not None and holding.get("totalShares") is not None and holding["totalShares"] > 0:
                        holding["currentTickerPrice"] = round(holding["currentValue"] / holding["totalShares"], 4)
                    if "cash reserve" in holding_name.lower() or "cash account" in holding_name.lower():
                        if holding.get("currentValue") is not None:
                            holding["currentTickerPrice"] = 1.00
                            holding["totalShares"] = holding["currentValue"]
                    holding_key = (holding_name.lower(), ticker.lower() if ticker else "")
                    if holding_key not in processed_holding_keys:
                        current_account_holdings.append(holding)
                        processed_holding_keys.add(holding_key)
                    else:
                        existing_holding = next(h for h in current_account_holdings if (clean_string(h.get("holdingName","")).lower(), clean_string(h.get("ticker","")).lower() if h.get("ticker") is not None else "") == holding_key)
                        existing_holding["totalShares"] = (existing_holding["totalShares"] or 0) + (holding["totalShares"] or 0)
                        existing_holding["currentValue"] = (existing_holding["currentValue"] or 0) + (holding["currentValue"] or 0)
                        logging.info(f"Consolidated duplicate holding: {holding_name} in account {account.get('accountId')}")
                account["holdings"] = current_account_holdings
                if account_key_base in unique_accounts_map:
                    existing_account = unique_accounts_map[account_key_base]
                    existing_account["currentValue"] = (existing_account.get("currentValue") or 0) + (account.get("currentValue") or 0)
                    existing_holdings_map = { (clean_string(h.get("holdingName","")).lower(), clean_string(h.get("ticker","")).lower() if h.get("ticker") is not None else ""): h for h in existing_account["holdings"] }
                    for new_holding in account["holdings"]:
                        new_holding_key = (clean_string(new_holding.get("holdingName","")).lower(), clean_string(new_holding.get("ticker","")).lower() if new_holding.get("ticker") is not None else "")
                        if new_holding_key in existing_holdings_map:
                            existing_h = existing_holdings_map[new_holding_key]
                            existing_h["totalShares"] = (existing_h["totalShares"] or 0) + (new_holding["totalShares"] or 0)
                            existing_h["currentValue"] = (existing_h["currentValue"] or 0) + (new_holding["currentValue"] or 0)
                        else:
                            existing_holdings_map[new_holding_key] = new_holding
                    existing_account["holdings"] = list(existing_holdings_map.values())
                    logging.info(f"Consolidated duplicate brokerage account: {account_key_base}")
                else:
                    unique_accounts_map[account_key_base] = account
            data["accounts"] = list(unique_accounts_map.values())
    elif doc_type == "pay":
        for tax in data.get("taxDeductions", []):
            raw_tax_type = clean_string(tax.get("taxType", "")).lower()
            if "federal" in raw_tax_type or "fed" in raw_tax_type:
                tax["taxType"] = "FEDERAL"
            elif "social security" in raw_tax_type or "fica" in raw_tax_type:
                tax["taxType"] = "SOCIAL_SECURITY"
            elif "medicare" in raw_tax_type:
                tax["taxType"] = "MEDICARE"
            elif "state" in raw_tax_type:
                tax["taxType"] = "STATE"
            elif "sdi" in raw_tax_type:
                tax["taxType"] = "SDI"
            elif "pfl" in raw_tax_type:
                tax["taxType"] = "PFL"
            elif "ltc" in raw_tax_type:
                tax["taxType"] = "LTC"
            elif "city" in raw_tax_type or "local" in raw_tax_type:
                tax["taxType"] = "CITY"
            else:
                tax["taxType"] = ""
        for ded in data.get("otherDeductions", []):
            raw_ded_type = clean_string(ded.get("deductionType", "")).lower()
            if "retirement" in raw_ded_type or "401k" in raw_ded_type or "403b" in raw_ded_type:
                ded["deductionType"] = "PRETAX_RETIREMENT"
            elif "medical" in raw_ded_type or "health" in raw_ded_type:
                ded["deductionType"] = "MEDICAL"
            elif "dental" in raw_ded_type:
                ded["deductionType"] = "DENTAL"
            elif "vision" in raw_ded_type:
                ded["deductionType"] = "VISION"
            elif "dependent" in raw_ded_type:
                ded["deductionType"] = "DEPENDENT_CARE"
            elif "fsa" in raw_ded_type:
                ded["deductionType"] = "FSA"
            elif "hsa" in raw_ded_type:
                ded["deductionType"] = "HSA"
            elif "roth" in raw_ded_type and "mega" in raw_ded_type:
                ded["deductionType"] = "ROTH_MEGA"
            elif "roth" in raw_ded_type:
                ded["deductionType"] = "ROTH"
            elif "espp" in raw_ded_type:
                ded["deductionType"] = "ESPP"
            elif "life" in raw_ded_type:
                ded["deductionType"] = "LIFE"
            elif "disability" in raw_ded_type:
                ded["deductionType"] = "DISABILITY"
            else:
                ded["deductionType"] = "OTHER"
            raw_ded_tax_type = clean_string(ded.get("deductionTaxType", "")).lower()
            if "pretax" in raw_ded_tax_type:
                ded["deductionTaxType"] = "PRETAX"
            elif "posttax" in raw_ded_tax_type:
                ded["deductionTaxType"] = "POSTTAX"
            else:
                ded["deductionTaxType"] = "OTHER"
        net_pay = data.get("paystubNetPay")
        if net_pay is not None and net_pay > 0:
            original_deductions_count = len(data.get("otherDeductions", []))
            data["otherDeductions"] = [
                d for d in data.get("otherDeductions", [])
                if d.get("deductionThisPeriod") is None or abs(d["deductionThisPeriod"] - net_pay) > 0.01
            ]
            if len(data["otherDeductions"]) < original_deductions_count:
                logging.info(f"Removed a deduction matching net pay for {state['document_id']}.")
    elif doc_type == "loan":
        current_loan_type = clean_string(data.get("type", "")).lower()
        if "mortgage" in current_loan_type or "home loan" in raw_text.lower():
            data["type"] = "Mortgages"
        elif "car" in current_loan_type or "auto" in current_loan_type or "vehicle loan" in raw_text.lower():
            data["type"] = "Car Loans"
            if data.get("escrow") is not None:
                if data["escrow"] == 0.0:
                    data["escrow"] = None
                elif data["escrow"] > 0 and not re.search(r"escrow", raw_text, re.IGNORECASE):
                    logging.warning(f"Escrow value {data['escrow']} found for car loan, but 'escrow' not mentioned in text. Setting to None.")
                    data["escrow"] = None
        elif "student" in current_loan_type or "education loan" in raw_text.lower():
            data["type"] = "Student Loans"
            if data.get("escrow") is not None:
                data["escrow"] = None
        else:
            data["type"] = "Other Loans"
            if data.get("escrow") is not None:
                data["escrow"] = None
    state["json_output"]["data"] = data
    logging.info(f"Post-processing complete for {state['document_id']}.")
    return state

async def evaluate_confidence(state: DocumentState) -> DocumentState:
    if not state["json_output"] or not state["json_output"].get("data"):
        state["genai_confidence_scores"] = {}
        state["decision_status"] = "rejected"
        state["overall_confidence_score"] = 0.0
        logging.warning(f"Confidence evaluation skipped: No data for {state['document_id']}")
        return state
    prompt = f"""
    Evaluate the confidence (0.0 to 1.0) of each extracted field in the provided JSON data from the document.
    Compare the parsed JSON data with the original document content (implied by the uploaded file, which is also provided to me) and the raw text provided below.
    Guidelines for scoring:
    - 1.0: Field value is directly and accurately extracted from the document with no ambiguity, and matches the document verbatim or with perfect numerical conversion/date formatting.
    - 0.9: Field value is accurately extracted but required minor cleaning, re-formatting, or direct derivation (e.g., calculating a simple total from obvious line items).
    - 0.8: Field value was accurately inferred (e.g., transaction category from description, specific loan type), or a sensible `null` or empty string was correctly assigned when the exact value was not explicit but contextually clear or not applicable.
    - 0.5: Field value was partially extracted, ambiguous, required significant inference/heuristics, or multiple possible interpretations existed. This includes cases where data might be consolidated from several sources but its original location is unclear.
    - 0.0: Field value is missing, incorrect, fabricated, or based on pure guesswork, or a required field is `null` when it should have been clearly present in the document.
    Return a JSON object where keys are flat dot-notation paths to the fields (e.g., 'employeeName': 1.0, 'accounts[0].holdings[0].currentValue': 0.9).
    Ensure ALL fields from the Extracted Data JSON are represented in the confidence scores.
    If an array contains multiple items (e.g., 'transactions', 'accounts', 'holdings'), provide a confidence score for EACH item and its sub-fields.
    Extracted Data:
    {json.dumps(state['json_output']['data'], indent=2, default=str)}
    Raw Text from Document (first 2000 characters for context, more might be used by the model):
    {state['raw_text'][:2000]}
    Output only the JSON object, wrapped in ```json\n...\n```.
    """
    confidence_scores = {}
    uploaded_file = None
    try:
        mime_type = get_mime_type(state["file_path"])
        uploaded_file = await retry_async_llm_call(asyncio.to_thread, genai.upload_file, state["file_path"], mime_type=mime_type)
        response = await retry_async_llm_call(asyncio.to_thread, model.generate_content, [prompt, uploaded_file])
        json_string = response.text.strip()
        json_match = re.search(r"```json\n([\s\S]*?)\n```", json_string)
        if json_match:
            json_payload = json_match.group(1)
            try:
                llm_confidence_raw = json.loads(json_payload)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse confidence scores JSON for {state['document_id']}: {e}. Attempting robust salvage. Response: {json_payload[:500]}...")
                llm_confidence_raw = _robust_json_parse(json_payload)
                if not llm_confidence_raw:
                    logging.error(f"Robust salvage for confidence scores also failed for {state['document_id']}.")
                    raise ValueError("Robust salvage for confidence scores failed.")
            def flatten_confidence_scores(obj, prefix="", flattened_dict=None):
                if flattened_dict is None:
                    flattened_dict = {}
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        if isinstance(value, (dict, list)):
                            flatten_confidence_scores(value, new_prefix, flattened_dict)
                        elif isinstance(value, (int, float)):
                            flattened_dict[new_prefix] = value
                        else:
                            logging.warning(f"Non-numeric confidence value at path {new_prefix}: {value} (type: {type(value)}) in {state['document_id']}")
                            flattened_dict[new_prefix] = 0.0
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        new_prefix = f"{prefix}[{i}]"
                        flatten_confidence_scores(item, new_prefix, flattened_dict)
                return flattened_dict
            confidence_scores = flatten_confidence_scores(llm_confidence_raw)
        else:
            logging.error(f"No JSON found in confidence response from LLM for {state['document_id']}. Raw LLM response: {json_string[:500]}...")
            raise ValueError("LLM did not return valid JSON for confidence evaluation.")
    except Exception as e:
        logging.error(f"Confidence evaluation failed for {state['document_id']}: {str(e)}. Assigning default low scores.")
        def assign_default_low_confidences_recursive(data_to_score, schema_path_prefix=""):
            if isinstance(data_to_score, dict):
                for key, value in data_to_score.items():
                    current_field_path = f"{schema_path_prefix}{key}"
                    assign_default_low_confidences_recursive(value, f"{current_field_path}.")
            elif isinstance(data_to_score, list):
                for i, item in enumerate(data_to_score):
                    assign_default_low_confidences_recursive(item, f"{schema_path_prefix}[{i}].")
            else:
                if schema_path_prefix and schema_path_prefix.endswith('.'):
                    field_key = schema_path_prefix[:-1]
                else:
                    field_key = schema_path_prefix
                actual_value = None
                try:
                    parts = field_key.replace(']', '').replace('[', '.').split('.')
                    current_val = state["json_output"]["data"]
                    for part in parts:
                        if part == "": continue
                        if isinstance(current_val, dict) and part in current_val:
                            current_val = current_val[part]
                        elif isinstance(current_val, list) and part.isdigit():
                            idx = int(part)
                            if len(current_val) > idx:
                                current_val = current_val[idx]
                            else:
                                current_val = None; break
                        else:
                            current_val = None; break
                    actual_value = current_val
                except Exception:
                    pass
                if actual_value is None or actual_value == "":
                    confidence_scores[field_key] = 0.0
                else:
                    confidence_scores[field_key] = 0.5
        assign_default_low_confidences_recursive(state["json_output"]["data"])
    finally:
        if uploaded_file:
            try:
                await asyncio.to_thread(genai.delete_file, uploaded_file.name)
            except Exception as e:
                logging.warning(f"Failed to delete uploaded file {uploaded_file.name}: {e}")
    state["genai_confidence_scores"] = confidence_scores
    return state

def calculate_final_decision_and_display(state: DocumentState) -> DocumentState:
    overall_confidence = 0.0
    if state["genai_confidence_scores"] and isinstance(state["genai_confidence_scores"], dict):
        numeric_scores = []
        for field, score in state["genai_confidence_scores"].items():
            if isinstance(score, (int, float)):
                numeric_scores.append(score)
            else:
                logging.error(f"Non-numeric confidence score found for field '{field}': {score} (Type: {type(score)}) in {state['document_id']}")
                numeric_scores.append(0.0)
        if numeric_scores:
            overall_confidence = sum(numeric_scores) / len(numeric_scores)
    else:
        overall_confidence = 0.0
    state["overall_confidence_score"] = overall_confidence
    print(f"\nOverall Confidence Score: {overall_confidence:.4f}")
    return state

def output_json(state: DocumentState) -> DocumentState:
    try:
        final_output_data = {
            "overall_confidence_score": round(state["overall_confidence_score"], 4),
            **state["json_output"]
        }
        print(f"\nCore data extraction by Gemini for {state['document_id']}:")
        print(json.dumps(final_output_data, indent=2, default=str))
        logging.info(f"Output displayed for {state['document_id']}")
    except Exception as e:
        logging.error(f"Error displaying JSON for {state['document_id']}: {str(e)}")
        print(f"Error displaying JSON for {state['document_id']}: {str(e)}")
    finally:
        if state["file_path"] and os.path.exists(state["file_path"]):
            try:
                os.remove(state["file_path"])
                logging.info(f"Cleaned up local file: {state['file_path']}")
            except Exception as e:
                logging.error(f"Error cleaning up {state['file_path']}: {str(e)}")
                print(f"Error cleaning up local file {state['file_path']}: {str(e)}")
    return state

# --- LangGraph Workflow Definition ---
workflow = StateGraph(DocumentState)
workflow.add_node("fetch_document", fetch_document)
workflow.add_node("classify_document", classify_document)
workflow.add_node("extract_pdf", extract_pdf)
workflow.add_node("parse_normalize", parse_normalize)
workflow.add_node("post_process_data", post_process_data)
workflow.add_node("evaluate_confidence", evaluate_confidence)
workflow.add_node("calculate_final_decision_and_display", calculate_final_decision_and_display)
workflow.add_node("output_json", output_json)
workflow.set_entry_point("fetch_document")
workflow.add_edge("fetch_document", "classify_document")
workflow.add_edge("classify_document", "extract_pdf")
workflow.add_edge("extract_pdf", "parse_normalize")
workflow.add_edge("parse_normalize", "post_process_data")
workflow.add_edge("post_process_data", "evaluate_confidence")
workflow.add_edge("evaluate_confidence", "calculate_final_decision_and_display")
workflow.add_edge("calculate_final_decision_and_display", "output_json")
graph = workflow.compile()

# --- Main Execution ---
async def main():
    print("Please upload a PDF, PNG, JPG, or JPEG file:")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded.")
        return
    for filename, content in uploaded.items():
        if not filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            print(f"Skipping {filename}: Only PDF, PNG, JPG, or JPEG files are supported.")
            continue
        temp_path = f"/tmp/{filename}"
        with open(temp_path, 'wb') as f:
            f.write(content)
        print(f"\n--- Processing '{filename}' ---")
        initial_state = DocumentState(
            file_path=temp_path,
            doc_type="",
            raw_text="",
            tables=[],
            json_output={},
            document_id=filename,
            ocr_confidences={},
            genai_confidence_scores={},
            decision_status="",
            overall_confidence_score=0.0
        )
        try:
            await graph.ainvoke(initial_state)
            print(f"\n--- Processing completed for {filename} ---")
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            print(f"Error processing {filename}: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logging.info(f"Cleaned up local file: {temp_path}")
                except Exception as e:
                    logging.error(f"Error cleaning up {temp_path}: {str(e)}")
                    print(f"Error cleaning up local file {temp_path}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())