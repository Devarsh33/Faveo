import sys
import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Prevents CUDA errors if running on CPU

import torch

model_path = os.path.join(os.getcwd(), "models", "flan_t5_base")

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
    "VM": ["remove host", "decommission vm", "track vm", "nessus"],
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
    input_text = "Summarize this ticket in a few sentences naturally: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(**inputs, max_length=50, min_length=10, length_penalty=1.5, do_sample=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Avoid redundancy: If summary is too similar to the body, rephrase it
    if summary.lower().strip() == text.lower().strip():
        summary = "This ticket requests assistance with: " + text[:150] + "..."  # Trim long descriptions

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
    """Process the ticket: summarize and assign a tag."""
    title = ticket.get("title", "").strip()
    body = ticket.get("body", "").strip()

    summary = nlp_summarize(body)
    tag = classify_ticket(title, summary)

    return {"summary": summary, "tag": tag}


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
