"""
g
marvin_mailer.py  —  Marvin Automatic Email Agent
===================================================
Fixes in this version
----------------------
1. Migrated to Cohere Chat API  (Generate API removed Sept 2025)
2. Auto-reply only runs on TODAY'S emails  (not all-time unread)
3. MIME-encoded subjects decoded correctly  (=?UTF-8?B?...?=)
4. Pandas sent-log created and written correctly
5. Sender name derived from Gmail address  (not hardcoded "Marvin")
6. Concurrent account processing via ThreadPoolExecutor
"""

import imaplib
import smtplib
import email
import email.utils
from email.header import decode_header as _decode_header
from email.message import EmailMessage
import datetime
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import dotenv_values

from email1 import generate_gmail   # returns parsed dict

# ── Config ────────────────────────────────────────────────────────────────────
env_vars = dotenv_values(".env")

ACCOUNTS = [
    {"user": env_vars.get("GMAIL_USER1"), "pass": env_vars.get("GMAIL_PASS1")},
    {"user": env_vars.get("GMAIL_USER2"), "pass": env_vars.get("GMAIL_PASS2")},
    # Add more accounts here
]

SENT_LOG_PATH  = "sent_log.csv"
LOG_LOCK       = threading.Lock()
REPLY_INTERVAL = 6 * 3600   # seconds between auto-reply cycles

# ── Decode MIME-encoded headers (e.g. =?UTF-8?B?...?=) ───────────────────────
def decode_subject(raw: str) -> str:
    """Safely decode any MIME-encoded email subject to a plain string."""
    if not raw:
        return "(No Subject)"
    parts = _decode_header(raw)
    decoded = []
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded).strip()

# ── Pandas sent-mail tracker ──────────────────────────────────────────────────
_LOG_COLS = ["timestamp", "account", "to_addr", "message_id", "subject"]

def _load_log() -> pd.DataFrame:
    if os.path.exists(SENT_LOG_PATH):
        df = pd.read_csv(SENT_LOG_PATH, dtype=str).fillna("")
        for col in _LOG_COLS:            # guarantee schema
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=_LOG_COLS)

def _append_log(account: str, to_addr: str, message_id: str, subject: str) -> None:
    """Append one row to sent_log.csv (thread-safe)."""
    row = pd.DataFrame([{
        "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
        "account":    account,
        "to_addr":    to_addr,
        "message_id": message_id,
        "subject":    subject,
    }])
    with LOG_LOCK:
        row.to_csv(
            SENT_LOG_PATH,
            mode="a",
            header=not os.path.exists(SENT_LOG_PATH),
            index=False,
        )

def already_replied(account: str, message_id: str) -> bool:
    """True if this Message-ID was already replied to from this account."""
    if not message_id or not os.path.exists(SENT_LOG_PATH):
        return False
    df = _load_log()
    if df.empty:
        return False
    return bool(((df["account"] == account) & (df["message_id"] == message_id)).any())

# ── IMAP helpers ──────────────────────────────────────────────────────────────
def _imap_connect(gmail_user: str, gmail_pass: str):
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    try:
        mail.login(gmail_user, gmail_pass)
        return mail
    except imaplib.IMAP4.error as exc:
        print(f"[ERROR] IMAP login failed for {gmail_user}: {exc}")
        return None

def fetch_today_emails(gmail_user: str, gmail_pass: str) -> list:
    """Fetch all emails received TODAY (used for both summary and auto-reply)."""
    mail = _imap_connect(gmail_user, gmail_pass)
    if not mail:
        return []
    mail.select("inbox")
    today   = datetime.datetime.now().strftime("%d-%b-%Y")
    _, msgs = mail.search(None, f'(ON "{today}")')
    ids     = msgs[0].split()
    emails  = []
    for eid in ids:
        _, msg_data = mail.fetch(eid, "(RFC822)")
        for part in msg_data:
            if isinstance(part, tuple):
                emails.append(email.message_from_bytes(part[1]))
    mail.logout()
    return emails

# ── SMTP sender ───────────────────────────────────────────────────────────────
def send_email(
    gmail_user: str, gmail_pass: str,
    to_addr: str, subject: str, body: str,
) -> None:
    msg            = EmailMessage()
    msg["From"]    = gmail_user
    msg["To"]      = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(gmail_user, gmail_pass)
        smtp.send_message(msg)

# ── Email utilities ───────────────────────────────────────────────────────────
def is_reply_needed(msg) -> bool:
    from_addr = msg.get("From", "").lower()
    return "no-reply" not in from_addr and "noreply" not in from_addr

def get_email_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode(errors="ignore")
    else:
        return msg.get_payload(decode=True).decode(errors="ignore")
    return ""

# ── Core auto-reply (one account, today's emails only) ───────────────────────
def auto_reply_to_emails(gmail_user: str, gmail_pass: str) -> None:
    # Derive a readable display name from the Gmail address
    sender_display = gmail_user.split("@")[0].replace(".", " ").title()

    # Only process TODAY's emails — never old/unread backlog
    emails = fetch_today_emails(gmail_user, gmail_pass)

    for msg in emails:
        if not is_reply_needed(msg):
            continue

        message_id     = msg.get("Message-ID", "").strip()
        raw_subject    = msg.get("Subject", "")
        subject        = decode_subject(raw_subject)          # ← MIME decode
        from_addr      = email.utils.parseaddr(msg.get("From", ""))[1]
        body           = get_email_body(msg)

        # Skip if we already replied to this exact message
        if already_replied(gmail_user, message_id):
            print(f"[SKIP] Already replied → {from_addr} | {subject}")
            continue

        print(f"[REPLY] {gmail_user} → {from_addr} | {subject}")

        # Generate + parse email in one call
        parsed = generate_gmail(
            prompt=(
                f"Reply to this email:\n"
                f"Subject: {subject}\n"
                f"Body: {body}"
            ),
            length="medium",
            sender_name=sender_display,
            recipient_name=from_addr,
        )

        reply_subject = parsed["subject"] or f"Re: {subject}"
        reply_body    = "\n".join(filter(None, [
            parsed["greeting"], "",
            parsed["body"],     "",
            parsed["closing"],
            parsed["signature"],
        ]))

        send_email(gmail_user, gmail_pass, from_addr, reply_subject, reply_body)

        _append_log(
            account=gmail_user,
            to_addr=from_addr,
            message_id=message_id,
            subject=reply_subject,
        )
        print(f"[DONE]  Replied + logged → {from_addr}")

# ── CLI commands ──────────────────────────────────────────────────────────────
def summarize_today_emails() -> None:
    for account in ACCOUNTS:
        user, pwd = account.get("user"), account.get("pass")
        if not user or not pwd:
            continue
        emails = fetch_today_emails(user, pwd)
        print(f"\n── {user} ── ({len(emails)} email(s) today)")
        if not emails:
            print("  No emails today.")
            continue
        for idx, msg in enumerate(emails, 1):
            from_addr = email.utils.parseaddr(msg.get("From", ""))[1]
            subject   = decode_subject(msg.get("Subject", ""))
            print(f"  {idx:>3}. {from_addr:<40} {subject}")

def show_sent_log() -> None:
    if not os.path.exists(SENT_LOG_PATH):
        print("No sent log yet — no replies have been sent.")
        return
    df = _load_log()
    if df.empty:
        print("Sent log is empty.")
        return
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 130)
    print(f"\n── Sent Log ({len(df)} entr{'y' if len(df)==1 else 'ies'}) ──")
    print(df.to_string(index=False))

# ── Auto-reply loop (all accounts run concurrently) ──────────────────────────
def _auto_reply_loop() -> None:
    print("[START] Auto-reply thread running.")
    while True:
        try:
            valid = [a for a in ACCOUNTS if a.get("user") and a.get("pass")]
            with ThreadPoolExecutor(max_workers=max(len(valid), 1)) as pool:
                futures = {
                    pool.submit(auto_reply_to_emails, a["user"], a["pass"]): a["user"]
                    for a in valid
                }
                for future in as_completed(futures):
                    acct = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"[ERROR] {acct}: {exc}")

            print(f"[CYCLE DONE] Sleeping {REPLY_INTERVAL // 3600}h until next cycle…")
            time.sleep(REPLY_INTERVAL)

        except Exception as exc:
            import traceback
            print(f"[CRASH] {exc}")
            traceback.print_exc()
            print("[RESTART] Restarting in 30 s…")
            time.sleep(30)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=_auto_reply_loop, daemon=True).start()

    COMMANDS = {"summary": summarize_today_emails, "log": show_sent_log}
    print("Commands: summary | log | exit\n")
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "exit":
            break
        elif cmd in COMMANDS:
            COMMANDS[cmd]()
        else:
            print("Unknown command. Try: summary | log | exit")
