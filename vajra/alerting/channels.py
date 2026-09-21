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


def _post_form(url: str, fields: dict, headers: dict | None = None,
               timeout: int = 15) -> tuple[int, str]:
    """application/x-www-form-urlencoded POST (Gupshup-style APIs)."""
    import urllib.parse
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(3000).decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


def _needs_unicode(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


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
    "whatsapp": None,   # wired below (Gupshup primary); generics kept too
    "sms": None,
    "ivr": None,
    "email": EmailChannel(),
    "whatsapp_meta": WhatsAppChannel(),
    "sms_generic": SMSGateway(),
    "ivr_generic": IVRChannel(),
}


# ---------------------------------------------------------------------------
# Provider implementations (Gupshup single-account stack for IN delivery)
# ---------------------------------------------------------------------------
class GupshupWhatsApp(Channel):
    """Gupshup WhatsApp: session text, or approved-template mode.

    Env: GUPSHUP_APIKEY, GUPSHUP_WA_SOURCE (E.164 business number; sandbox
    philosopher's proxy is 917834811114), GUPSHUP_WA_APP (src.name).
    Optional template mode: GUPSHUP_WA_TEMPLATE_ID (+ LANG, default en) where
    the approved template takes {{1}} = full alert text.
    Sandbox note: the recipient must first opt in by messaging the sandbox
    number; session text works inside the 24-h window.
    """
    name = "whatsapp"
    ENDPOINT = "https://api.gupshup.io/wa/api/v1/msg"
    TEMPLATE_ENDPOINT = "https://api.gupshup.io/wa/api/v1/template/msg"

    def available(self) -> tuple[bool, str]:
        ok = bool(os.environ.get("GUPSHUP_APIKEY")
                  and os.environ.get("GUPSHUP_WA_SOURCE")
                  and os.environ.get("GUPSHUP_WA_APP"))
        return (True, "ok") if ok else (
            False, "GUPSHUP_APIKEY/GUPSHUP_WA_SOURCE/GUPSHUP_WA_APP unset")

    def send(self, to: str, text: str, meta: dict) -> dict:
        ok, _ = self.available()
        if not ok or not to:
            return FileChannel().send(to, "[whatsapp-stub] " + text, meta) | \
                {"channel": "whatsapp", "status": "queued-no-credentials"}
        tid = os.environ.get("GUPSHUP_WA_TEMPLATE_ID", "")
        headers = {"apikey": os.environ["GUPSHUP_APIKEY"]}
        if tid:
            code, body = _post_form(self.TEMPLATE_ENDPOINT, {
                "channel": "whatsapp",
                "source": os.environ["GUPSHUP_WA_SOURCE"],
                "destination": to,
                "src.name": os.environ["GUPSHUP_WA_APP"],
                "template": json.dumps({
                    "id": tid, "params": [text[:900]],
                    "language": os.environ.get("GUPSHUP_WA_LANG", "en")}),
            }, headers)
        else:
            code, body = _post_form(self.ENDPOINT, {
                "channel": "whatsapp",
                "source": os.environ["GUPSHUP_WA_SOURCE"],
                "destination": to,
                "src.name": os.environ["GUPSHUP_WA_APP"],
                "message": json.dumps({"type": "text", "text": text[:1000]}),
            }, headers)
        if code == 200 and '"status":"submitted"' in body.replace(" ", ""):
            return _receipt("whatsapp", to, "delivered", body[:160])
        return _receipt("whatsapp", to, "failed", f"http={code} {body[:160]}")


class GupshupSMS(Channel):
    """Gupshup Enterprise SMS (GatewayAPI). Unicode_text auto-selected for
    Hindi/Marathi. Env: GUPSHUP_SMS_USERID, GUPSHUP_SMS_PASSWORD.
    India note: DLT entity + sender-ID/template registration is required by
    regulation for production traffic; sandbox testing works without it.
    """
    name = "sms"
    ENDPOINT = "https://enterprise.smsgupshup.com/GatewayAPI/rest"

    def available(self) -> tuple[bool, str]:
        ok = bool(os.environ.get("GUPSHUP_SMS_USERID")
                  and os.environ.get("GUPSHUP_SMS_PASSWORD"))
        return (True, "ok") if ok else (
            False, "GUPSHUP_SMS_USERID/GUPSHUP_SMS_PASSWORD unset")

    def send(self, to: str, text: str, meta: dict) -> dict:
        ok, _ = self.available()
        if not ok or not to:
            return FileChannel().send(to, "[sms-stub] " + text, meta) | \
                {"channel": "sms", "status": "queued-no-credentials"}
        code, body = _post_form(self.ENDPOINT, {
            "method": "sendMessage", "v": "1.1", "auth_scheme": "plain",
            "format": "json",
            "userid": os.environ["GUPSHUP_SMS_USERID"],
            "password": os.environ["GUPSHUP_SMS_PASSWORD"],
            "send_to": to.replace("+", "").replace(" ", ""),
            "msg_type": "Unicode_text" if _needs_unicode(text) else "text",
            "msg": text[:459],
        })
        if code == 200 and "success" in body.lower():
            return _receipt("sms", to, "delivered", body[:160])
        return _receipt("sms", to, "failed", f"http={code} {body[:160]}")


class ExotelIVR(Channel):
    """Exotel click-to-call with optional TTS flow URL.

    Env: EXOTEL_SID, EXOTEL_TOKEN, EXOTEL_CALLERID (Exotel virtual number),
    optional EXOTEL_FLOW_URL (voice applet that Says the alert text).
    Without a flow URL the call connects the recipient to the caller ID
    (callback pattern); with it, the applet speaks the alert.
    """
    name = "ivr"

    def available(self) -> tuple[bool, str]:
        ok = bool(os.environ.get("EXOTEL_SID")
                  and os.environ.get("EXOTEL_TOKEN")
                  and os.environ.get("EXOTEL_CALLERID"))
        return (True, "ok") if ok else (
            False, "EXOTEL_SID/EXOTEL_TOKEN/EXOTEL_CALLERID unset")

    def send(self, to: str, text: str, meta: dict) -> dict:
        ok, _ = self.available()
        if not ok or not to:
            return FileChannel().send(to, "[ivr-stub] " + text, meta) | \
                {"channel": "ivr", "status": "queued-no-credentials"}
        import base64
        sid = os.environ["EXOTEL_SID"]
        token = os.environ["EXOTEL_TOKEN"]
        cred = base64.b64encode(f"{sid}:{token}".encode()).decode()
        fields = {"From": to.replace("+", "").replace(" ", ""),
                  "CallerId": os.environ["EXOTEL_CALLERID"],
                  "CallType": "trans"}
        if os.environ.get("EXOTEL_FLOW_URL"):
            fields["Url"] = os.environ["EXOTEL_FLOW_URL"]
        code, body = _post_form(
            f"https://api.exotel.com/v1/Accounts/{sid}/Calls/connect.json",
            fields, {"Authorization": f"Basic {cred}"})
        if code in (200, 201) and "queued" in body.lower():
            return _receipt("ivr", to, "delivered", body[:160])
        return _receipt("ivr", to, "failed", f"http={code} {body[:160]}")


CHANNELS.update({
    "whatsapp": GupshupWhatsApp(),
    "sms": GupshupSMS(),
    "ivr": ExotelIVR(),
})
