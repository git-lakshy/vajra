"""Interactive alert-channel setup: configure credentials + live-test each.

  python setup_alerts.py           # guided setup for every channel
  python setup_alerts.py --check   # report live/stub status, no prompts

Writes .env (gitignored, never committed). Server and drill auto-load it.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from vajra.envfile import load_dotenv, write_dotenv  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def ask(prompt: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        v = input(f"  {prompt}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    return v or default


def test_send(channel: str, to: str, text: str) -> dict:
    from vajra.alerting.channels import CHANNELS
    return CHANNELS[channel].send(to, text, {"ts": "setup-test",
                                             "role": "setup",
                                             "hazard": "test",
                                             "poi": "setup"})


def setup_telegram() -> None:
    print("\n[1/5] Telegram (free, instant - best first live channel)")
    print("  a) Open Telegram, message @BotFather -> /newbot -> copy the token")
    print("  b) Message your bot once (any text), then open")
    print("     https://api.telegram.org/bot<TOKEN>/getUpdates to find chat.id")
    tok = ask("Bot token (blank = skip)")
    if not tok:
        return
    os.environ["VAJRA_TELEGRAM_TOKEN"] = tok
    chat = ask("Your chat ID (digits)")
    r = test_send("telegram", chat, "VAJRA setup test - Telegram live ✔ (demo)")
    print(f"  -> {r['status']}: {r['detail'][:100]}")
    if r["status"] == "delivered":
        write_dotenv(".env", {"VAJRA_TELEGRAM_TOKEN": tok})
        print("  saved to .env")


def setup_email() -> None:
    print("\n[2/5] Email via Gmail SMTP (free, instant)")
    print("  Needs a Google App Password: Google Account -> Security -> 2-Step")
    print("  Verification ON -> App passwords -> create one (16 letters).")
    user = ask("Gmail address (blank = skip)")
    if not user:
        return
    pw = ask("App password (16 letters, no spaces needed)")
    os.environ.update({"VAJRA_SMTP_HOST": "smtp.gmail.com",
                       "VAJRA_SMTP_PORT": "587",
                       "VAJRA_SMTP_USER": user,
                       "VAJRA_SMTP_PASS": pw.replace(" ", ""),
                       "VAJRA_SMTP_FROM": user})
    r = test_send("email", user, "VAJRA setup test - email live (demo).")
    print(f"  -> {r['status']}: {r['detail'][:100]}")
    if r["status"] == "delivered":
        write_dotenv(".env", {"VAJRA_SMTP_HOST": "smtp.gmail.com",
                              "VAJRA_SMTP_PORT": "587",
                              "VAJRA_SMTP_USER": user,
                              "VAJRA_SMTP_PASS": pw.replace(" ", ""),
                              "VAJRA_SMTP_FROM": user})
        print("  saved to .env")


def setup_gupshup_wa() -> None:
    print("\n[3/5] WhatsApp via Gupshup (one account covers WA+SMS)")
    print("  a) Sign up at gupshup.io -> create a WhatsApp app (sandbox is free)")
    print("  b) Sandbox: use proxy source 917834811114 and opt in by messaging it")
    print("  c) Dashboard -> app -> API key; note source number + app name")
    key = ask("Gupshup API key (blank = skip)")
    if not key:
        return
    os.environ["GUPSHUP_APIKEY"] = key
    src = ask("Source number E.164, sandbox = 917834811114",
              "917834811114")
    app = ask("App name (src.name)")
    to = ask("Your WhatsApp number E.164 (e.g. 9198XXXXXXXX)")
    os.environ.update({"GUPSHUP_WA_SOURCE": src, "GUPSHUP_WA_APP": app})
    r = test_send("whatsapp", to, "VAJRA setup test - WhatsApp live (demo).")
    print(f"  -> {r['status']}: {r['detail'][:120]}")
    if r["status"] == "delivered":
        write_dotenv(".env", {"GUPSHUP_APIKEY": key,
                              "GUPSHUP_WA_SOURCE": src,
                              "GUPSHUP_WA_APP": app})
        print("  saved to .env")
    else:
        print("  NOT saved (fix the error above and re-run). "
              "Common: recipient not opted into the sandbox.")


def setup_gupshup_sms() -> None:
    print("\n[4/5] SMS via Gupshup Enterprise (India: DLT needed for production)")
    print("  Dashboard -> Enterprise SMS -> userid + password.")
    print("  Production India traffic needs DLT entity + sender ID + templates;")
    print("  sandbox/testing works without DLT.")
    uid = ask("Gupshup SMS userid (blank = skip)")
    if not uid:
        return
    pw = ask("Gupshup SMS password")
    to = ask("Your mobile E.164 (e.g. 9198XXXXXXXX)")
    os.environ.update({"GUPSHUP_SMS_USERID": uid,
                       "GUPSHUP_SMS_PASSWORD": pw})
    r = test_send("sms", to, "VAJRA setup test (demo).")
    print(f"  -> {r['status']}: {r['detail'][:120]}")
    if r["status"] == "delivered":
        write_dotenv(".env", {"GUPSHUP_SMS_USERID": uid,
                              "GUPSHUP_SMS_PASSWORD": pw})
        print("  saved to .env")
    else:
        print("  NOT saved (fix the error above and re-run).")


def setup_exotel() -> None:
    print("\n[5/5] IVR voice calls via Exotel")
    print("  Exotel dashboard -> API key (SID) + API token + an Exotel number.")
    print("  Optional: an Exotel flow URL whose applet Says the alert text.")
    sid = ask("Exotel SID (blank = skip)")
    if not sid:
        return
    tok = ask("Exotel API token")
    caller = ask("Exotel caller ID (your Exotel number, E.164)")
    flow = ask("Flow URL with Say applet (optional, blank = callback only)")
    to = ask("Your mobile E.164 to receive the test call")
    env = {"EXOTEL_SID": sid, "EXOTEL_TOKEN": tok,
           "EXOTEL_CALLERID": caller}
    if flow:
        env["EXOTEL_FLOW_URL"] = flow
    os.environ.update(env)
    print("  placing test call (answer to verify)...")
    r = test_send("ivr", to, "VAJRA setup test call. This is a demo.")
    print(f"  -> {r['status']}: {r['detail'][:120]}")
    if r["status"] == "delivered":
        write_dotenv(".env", env)
        print("  saved to .env")
    else:
        print("  NOT saved (fix the error above and re-run).")


def check() -> None:
    load_dotenv()
    from vajra.alerting.channels import CHANNELS
    print("channel status (from env/.env):")
    for name, ch in CHANNELS.items():
        ok, reason = ch.available()
        print(f"  {name:15s} {'LIVE' if ok else 'stub':5s}  {reason}")


def main() -> None:
    ap = argparse.ArgumentParser(description="VAJRA alert channel setup")
    ap.add_argument("--check", action="store_true",
                    help="report status without prompting")
    args = ap.parse_args()
    load_dotenv()
    if args.check:
        return check()
    print("VAJRA alert setup - keys go to .env (gitignored, never committed).")
    print("Skip anything with a blank answer; re-run anytime.")
    setup_telegram()
    setup_email()
    setup_gupshup_wa()
    setup_gupshup_sms()
    setup_exotel()
    print("\nDone. Final status:")
    check()
    print("\nNext: register recipients with "
          "POST /api/subscribers/{id}  e.g. "
          '{"telegram": "<chat_id>", "sms": "+91...", "languages": ["hi"]}')


if __name__ == "__main__":
    main()
