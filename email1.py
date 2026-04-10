import cohere
import re
from dotenv import dotenv_values

# ── Load env ──────────────────────────────────────────────────────────────────
env_vars     = dotenv_values(".env")
CohereAPIKey = env_vars.get("CohereAPIKey")

# Single shared client – instantiated once at import
co = cohere.ClientV2(api_key=CohereAPIKey)

# Pre-compiled keyword patterns for O(1) email-type classification
_TYPE_RULES = [
    (re.compile(r"\b(cold|introduce)\b"),       "cold"),
    (re.compile(r"\b(promot|offer|business)\b"), "promotional"),
    (re.compile(r"\b(reply|respond)\b"),         "reply"),
    (re.compile(r"\bfollow[\s-]?up\b"),          "followup"),
    (re.compile(r"\bthank\b"),                   "thankyou"),
    (re.compile(r"\b(apolog|sorry)\b"),          "apology"),
    (re.compile(r"\b(informal|friend)\b"),       "informal"),
    (re.compile(r"\b(formal|official)\b"),       "formal"),
]

# ── Email-type classifier ─────────────────────────────────────────────────────
def decide_email_type(prompt: str) -> str:
    p = prompt.lower()
    for pattern, label in _TYPE_RULES:
        if pattern.search(p):
            return label
    return "general"

# ── Parser ────────────────────────────────────────────────────────────────────
def parse_generated_email(email_text: str) -> dict:
    """
    Parse raw LLM output into { subject, greeting, body, closing, signature }.
    All missing fields default to empty string.
    """
    result = {"subject": "", "greeting": "", "body": "", "closing": "", "signature": ""}
    lines  = [ln.rstrip() for ln in email_text.splitlines() if ln.strip()]

    if not lines:
        return result

    # Subject — must be the first line
    if lines[0].lower().startswith("subject:"):
        result["subject"] = lines[0][len("subject:"):].strip()
        lines = lines[1:]

    if not lines:
        return result

    # Greeting — next non-empty line
    result["greeting"] = lines[0]
    lines = lines[1:]

    # Closing + Signature — last two lines; rest is body
    if len(lines) >= 2:
        result["closing"]   = lines[-2]
        result["signature"] = lines[-1]
        result["body"]      = "\n".join(lines[:-2]).strip()
    elif len(lines) == 1:
        result["body"] = lines[0]

    return result

# ── LLM call — Cohere Chat API (Generate API removed Sept 2025) ───────────────
def generate_gmail(
    prompt: str,
    length: str = "medium",
    sender_name: str = "Marvin",
    recipient_name: str = "Recipient",
    extra_instructions: str | None = None,
) -> dict:
    """
    Generate an email via Cohere Chat API.
    Returns a parsed dict: { subject, greeting, body, closing, signature }
    """
    email_type = decide_email_type(prompt)
    max_tok    = {"long": 800, "medium": 400, "short": 200}.get(length, 400)

    extra_line = (
        f"\nAdditional instructions: {extra_instructions}"
        if extra_instructions else ""
    )

    system_msg = (
        f"You are Marvin, an expert AI email writer.\n"
        f"Write a {email_type} email in perfect English.\n"
        f"Sender: {sender_name} | Recipient: {recipient_name}\n"
        f"Email type: {email_type} | Length: {length}\n"
        f"STRICT FORMAT RULES — follow exactly:\n"
        f"  Line 1: Subject: <subject text>\n"
        f"  Line 2: Greeting (e.g. Dear ...)\n"
        f"  Lines 3+: Body paragraphs\n"
        f"  Second-to-last line: Closing (e.g. Best regards,)\n"
        f"  Last line: Signature (sender name)\n"
        f"- Correct grammar, spelling and punctuation throughout.\n"
        f"{extra_line}"
    )

    response = co.chat(
        model="command-a-03-2025",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tok,
        temperature=0.7,
    )

    raw_text = response.message.content[0].text.strip()
    if not raw_text:
        raise ValueError("Cohere returned an empty response.")

    return parse_generated_email(raw_text)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Ultimate Marvin Gmail Generator\n")
    while True:
        user_prompt = input("Describe the email you want Marvin to write:\n> ")
        length      = input("Length (short/medium/long) [medium]: ").strip().lower() or "medium"
        sender      = input("Your name [Marvin]: ").strip() or "Marvin"
        recipient   = input("Recipient's name [Recipient]: ").strip() or "Recipient"
        extra       = input("Extra instructions? (optional): ").strip() or None

        parsed = generate_gmail(
            prompt=user_prompt, length=length,
            sender_name=sender, recipient_name=recipient,
            extra_instructions=extra,
        )
        print("\n--- Generated Email ---")
        print(f"Subject  : {parsed['subject']}")
        print(f"Greeting : {parsed['greeting']}")
        print(f"Body     :\n{parsed['body']}")
        print(f"Closing  : {parsed['closing']}")
        print(f"Signature: {parsed['signature']}")
        print("----------------------\n")
