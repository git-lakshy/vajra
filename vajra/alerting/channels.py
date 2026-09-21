"""Alert delivery channels: one interface, many transports.

Real-now (no approval needed beyond a token):
  telegram  - Telegram Bot API via plain HTTPS (needs VAJRA_TELEGRAM_TOKEN;
              chat IDs come from the subscriber directory)
  cap       - CAP 1.2 XML payload (returned for the CAP feed / SACHET path)
  file      - demo outbox on disk (logs/outbox/)
Provider hooks (activate with env credentials, else file-stubbed):
  whatsapp  - Meta WhatsApp Cloud API (VAJRA_WA_TOKEN, VAJRA_WA_PHONE_ID)
  sms       - generic HTTP SMS gateway (VAJRA_SMS_URL/KEY/FROM)
  ivr       - generic voice-call API (VAJRA_IVR_URL/KEY)
  email     - SMTP (VAJRA_SMTP_HOST/USER/PASS/FROM)
"""

from __future__ import annotations

import json
import os
import urllib.request


class Channel:
    name = "base"

    def available(self) -> tuple[bool, str]:
        """(ok, reason)."""
        return True, "ok"

    def send(self, to: str, text: str, meta: dict) -> dict:
        """Return a receipt: {channel, to, status, detail}."""
        raise NotImplementedError


def _receipt(channel: str, to: str, status: str, detail: str = "") -> dict:
    return {"channel": channel, "to": to, "status": status, "detail": detail}


def _post_json(url: str, payload: dict, headers: dict | None = None,
               timeout: int = 15) -> tuple[int, str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json",
                                          **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(3000).decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001 - transport errors become receipts
        return 0, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
class FileChannel(Channel):
    """Demo outbox: every message lands in logs/outbox/ as JSON."""
    name = "file"

    def send(self, to: str, text: str, meta: dict) -> dict:
        os.makedirs("logs/outbox", exist_ok=True)
        rec = {"to": to, "text": text, "meta": meta}
        fn = (f"logs/outbox/{meta.get('ts', 'notime').replace(':', '')}"
              f"_{meta.get('role', 'x')}_{meta.get('hazard', 'x')}.json")
        try:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
            return _receipt("file", to, "delivered", fn)
        except Exception as e:
            return _receipt("file", to, "failed", str(e))


class TelegramChannel(Channel):
    """Live Telegram delivery (Bot API, no extra dependencies)."""
    name = "telegram"

    def available(self) -> tuple[bool, str]:
        tok = os.environ.get("VAJRA_TELEGRAM_TOKEN", "")
        return (True, "ok") if tok else (False, "VAJRA_TELEGRAM_TOKEN unset")

    def send(self, to: str, text: str, meta: dict) -> dict:
        tok = os.environ.get("VAJRA_TELEGRAM_TOKEN", "")
        if not tok or not to:
            return _receipt("telegram", to, "skipped",
                            "no token or chat_id")
        code, body = _post_json(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            {"chat_id": to, "text": text})
        if code == 200 and '"ok":true' in body.replace(" ", ""):
            return _receipt("telegram", to, "delivered", body[:120])
        return _receipt("telegram", to, "failed", f"http={code} {body[:120]}")


class WhatsAppChannel(Channel):
    """Meta WhatsApp Cloud API (template text). Stubbed to file without creds."""
    name = "whatsapp"

    def available(self) -> tuple[bool, str]:
        ok = bool(os.environ.get("VAJRA_WA_TOKEN")
                  and os.environ.get("VAJRA_WA_PHONE_ID"))
        return (True, "ok") if ok else (
            False, "VAJRA_WA_TOKEN/VAJRA_WA_PHONE_ID unset")

    def send(self, to: str, text: str, meta: dict) -> dict:
        ok, _ = self.available()
        if not ok or not to:
            return FileChannel().send(to, "[whatsapp-stub] " + text, meta) | \
                {"channel": "whatsapp", "status": "queued-no-credentials"}
        code, body = _post_json(
            f"https://graph.facebook.com/v20.0/"
            f"{os.environ['VAJRA_WA_PHONE_ID']}/messages",
            {"messaging_product": "whatsapp", "to": to,
             "type": "text", "text": {"body": text[:1000]}},
            {"Authorization": f"Bearer {os.environ['VAJRA_WA_TOKEN']}"})
        if code == 200:
            return _receipt("whatsapp", to, "delivered", body[:120])
        return _receipt("whatsapp", to, "failed", f"http={code} {body[:120]}")


class SMSGateway(Channel):
    """Generic HTTP SMS gateway (MSG91/Gupshup-style). Stubbed w/o creds."""
    name = "sms"

    def available(self) -> tuple[bool, str]:
        ok = bool(os.environ.get("VAJRA_SMS_URL")
                  and os.environ.get("VAJRA_SMS_KEY"))
        return (True, "ok") if ok else (
            False, "VAJRA_SMS_URL/VAJRA_SMS_KEY unset")

    def send(self, to: str, text: str, meta: dict) -> dict:
        ok, _ = self.available()
        if not ok or not to:
            return FileChannel().send(to, "[sms-stub] " + text, meta) | \
                {"channel": "sms", "status": "queued-no-credentials"}
        code, body = _post_json(os.environ["VAJRA_SMS_URL"], {
            "api_key": os.environ["VAJRA_SMS_KEY"],
            "from": os.environ.get("VAJRA_SMS_FROM", "VAJRA"),
            "to": to, "text": text[:459]})
        if code == 200:
            return _receipt("sms", to, "delivered", body[:120])
        return _receipt("sms", to, "failed", f"http={code} {body[:120]}")


class IVRChannel(Channel):
    """Generic voice-call API (text-to-speech call). Stubbed w/o creds."""
    name = "ivr"

    def available(self) -> tuple[bool, str]:
        ok = bool(os.environ.get("VAJRA_IVR_URL")
                  and os.environ.get("VAJRA_IVR_KEY"))
        return (True, "ok") if ok else (
            False, "VAJRA_IVR_URL/VAJRA_IVR_KEY unset")

    def send(self, to: str, text: str, meta: dict) -> dict:
        ok, _ = self.available()
        if not ok or not to:
            return FileChannel().send(to, "[ivr-stub] " + text, meta) | \
                {"channel": "ivr", "status": "queued-no-credentials"}
        code, body = _post_json(os.environ["VAJRA_IVR_URL"], {
            "api_key": os.environ["VAJRA_IVR_KEY"], "to": to, "text": text})
        if code == 200:
            return _receipt("ivr", to, "delivered", body[:120])
        return _receipt("ivr", to, "failed", f"http={code} {body[:120]}")


class EmailChannel(Channel):
    """SMTP email. Stubbed to file without creds."""
    name = "email"

    def available(self) -> tuple[bool, str]:
        ok = bool(os.environ.get("VAJRA_SMTP_HOST")
                  and os.environ.get("VAJRA_SMTP_USER"))
        return (True, "ok") if ok else (
            False, "VAJRA_SMTP_HOST/VAJRA_SMTP_USER unset")

    def send(self, to: str, text: str, meta: dict) -> dict:
        ok, _ = self.available()
        if not ok or not to:
            return FileChannel().send(to, "[email-stub] " + text, meta) | \
                {"channel": "email", "status": "queued-no-credentials"}
        try:
            import smtplib
            from email.message import EmailMessage
            msg = EmailMessage()
            msg["Subject"] = (f"VAJRA {meta.get('hazard', '')} alert "
                              f"{meta.get('poi', '')}")
            msg["From"] = os.environ.get("VAJRA_SMTP_FROM",
                                         os.environ["VAJRA_SMTP_USER"])
            msg["To"] = to
            msg.set_content(text)
            with smtplib.SMTP(os.environ["VAJRA_SMTP_HOST"],
                              int(os.environ.get("VAJRA_SMTP_PORT", "587")),
                              timeout=15) as s:
                s.starttls()
                s.login(os.environ["VAJRA_SMTP_USER"],
                        os.environ.get("VAJRA_SMTP_PASS", ""))
                s.send_message(msg)
            return _receipt("email", to, "delivered", "smtp ok")
        except Exception as e:
            return _receipt("email", to, "failed", f"{type(e).__name__}: {e}")


CHANNELS: dict[str, Channel] = {
    "file": FileChannel(),
    "telegram": TelegramChannel(),
    "whatsapp": WhatsAppChannel(),
    "sms": SMSGateway(),
    "ivr": IVRChannel(),
    "email": EmailChannel(),
}
