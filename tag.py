import sys
import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Prevents CUDA errors if running on CPU

import torch

# script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
model_path = "google/flan-t5-base"

# Load Flan-T5 Base model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Ticket tags with relevant keywords
tags_with_keywords = {
    "IPWhitelisting": ["ip whitelist", "allow ip", "okta security networks", "threat defense"],
    "AppLocker": ["applocker", "application execution", "block application"],
    "ADSecurityGroup": ["ad security group", "exchange group", "remove user from group", "active directory"],
    "AttachmentRelease": ["release attachment", "password protected attachment", "allow email attachment"],
    "EmailWhitelisting": ["email whitelist", "allow email", "proofpoint", "deliver email"],
    "AppAssignment": ["application assignment", "add user to app", "app access"],
    "chatops:mfa-bypass": ["mfa bypass", "temporary mfa", "bypass mfa"],
    "VM": ["remove host", "decommission vm", "track vm", "nessus", "remove server", "tenable", "decom"],
    "PhishingReport": ["phishing report", "possible phishing", "malicious email"],
    "GenericInformation": ["generic ticket", "secops ticket", "general request"],
    "PasswordReset": ["password reset", "okta reset", "ad password reset"],
    "KeeperAccounts": ["keeper password manager", "keeper reset", "keeper issue"],
    "MemberIssues": ["member account issue", "security concern", "account investigation"],
    "ADPassword": ["ad password issue", "active directory password reset"],
    "OktaMFAResets": ["okta mfa reset", "reset multi-factor"],
    "Zscaler": ["zscaler issue", "zscaler request", "zscaler domain allow", "zscaler", "zscalar", "zscaler domain block", "domain block"],
    "chatops:cs-usb": ["usb whitelist", "allow usb", "cs-usb"],
    "USBDeviceControl": ["usb device request", "usb security", "block usb"],
    "Imperva": ["imperva issue", "imperva cdn"],
    "AccessRequest": ["access request", "permission request", "grant access"],
}


def nlp_summarize(text):
    """Perform natural-language summarization for better readability."""
    input_text = (
        "You are an AI model trained to process IT security support tickets. Your task is to summarize the following "
        "ticket while preserving essential security-related details.\n\n"
        "### Instructions:\n"
        "1. Identify the core issue or request described in the ticket.\n"
        "2. Extract relevant technical details, including affected applications, services, security tools, and account names if mentioned.\n"
        "3. Identify requested actions, such as whitelisting, password resets, access approvals, or investigations.\n"
        "4. Retain important security context, including any references to phishing, multi-factor authentication (MFA), VPN, Active Directory, or security tools.\n"
        "5. Be concise: Keep the summary within 1-2 sentences while ensuring that no critical details are lost.\n"
        "6. Avoid unnecessary words: Do not include greetings, apologies, or generic phrases unless they add meaningful context.\n"
        "7. Preserve clarity: Ensure the summary is actionable and easy to understand by a cybersecurity analyst.\n\n"
        "### Ticket Information:\n"
        f"{text}\n\n"
        "### Expected Summary Format:\n"
        "- Clearly state the problem and requested action.\n"
        "- Mention the affected system/service/tool if specified.\n"
        "- Avoid generic or vague responses."
    )
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    summary_ids = model.generate(
        **inputs,
        max_length=40,
        min_length=10,
        length_penalty=1.0,
        repetition_penalty=1.5,
        num_beams=3,
        early_stopping=True,
        do_sample=False,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    # If summary is too vague or unhelpful, fallback to trimmed body
    if len(summary.split()) < 5:
        summary = "This ticket is about: " + text[:100] + "..."

    return summary


def classify_ticket(title, summary):
    """Classify ticket based on title, summary, and tag keywords."""
    full_text = title.lower() + " " + summary.lower()

    # Check for best matching tag
    matched_tag = None
    for tag, keywords in tags_with_keywords.items():
        if any(keyword in full_text for keyword in keywords):
            matched_tag = tag
            break  # Stop at first accurate match

    return matched_tag if matched_tag else "GenericInformation"


def process_ticket(ticket):
    """Process the ticket: summarize and assign a tag while preserving token and ID."""
    title = ticket.get("title", "")
    body = ticket.get("body", "")
    ticket_id = ticket.get("id", None)  # Preserve ID if available

    summary = nlp_summarize(body)
    tag = classify_ticket(title, summary)

    return {"id": ticket_id, "summary": summary, "tag": tag}


if __name__ == "__main__":
    try:
        # with open("input.json", "r") as f:
        #     ticket = json.load(f)

        # Read input from Tines (via stdin)
        input_data = sys.stdin.read()
        ticket = json.loads(input_data)

        # Process the ticket
        result = process_ticket(ticket)

        # Print output in JSON format
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(json.dumps({"error": str(e)}))