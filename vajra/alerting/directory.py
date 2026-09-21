"""Subscriber directory + multilingual message templates.

Directory entries are demo-seeded (no real PII). In deployment, subscriber
management (opt-in, chat IDs, phone numbers) lives behind /api/subscribers.
Languages: en (English), hi (Hindi), mr (Marathi) - short, SMS-safe texts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Subscriber:
    id: str
    role: str                       # district | aviation | agriculture
    poi: str                        # POI name or "__domain__"
    languages: tuple[str, ...] = ("en",)
    channels: dict[str, str] = field(default_factory=dict)
    # channels maps channel -> address, e.g. {"telegram": "<chat_id>",
    # "sms": "+91...", "whatsapp": "+91...", "email": "a@b.c"}


def seed_directory(pois: list[str]) -> list[Subscriber]:
    """Demo directory: one contact per role per POI (addresses empty until
    registered via the API)."""
    subs = []
    for poi in pois:
        subs.append(Subscriber(f"dm-{poi}", "district", poi, ("en", "hi")))
        subs.append(Subscriber(f"atc-{poi}", "aviation", poi, ("en",)))
        subs.append(Subscriber(f"agri-{poi}", "agriculture", poi,
                               ("en", "hi", "mr")))
    return subs


# ---------------------------------------------------------------------------
# Templates: {hazard_word, poi, eta, prob, action}
# ---------------------------------------------------------------------------
_T = {
    "hail": {
        "word": {"en": "HAIL", "hi": "ओले", "mr": "गारपीट"},
        "action": {
            "en": "Move indoors. Protect livestock and vehicles.",
            "hi": "घर के अंदर रहें। पशुओं और वाहनों की रक्षा करें।",
            "mr": "घरात रहा। जनावरे व वाहने सुरक्षित ठेवा।",
        },
    },
    "lightning": {
        "word": {"en": "LIGHTNING", "hi": "बिजली", "mr": "वीज"},
        "action": {
            "en": "Move indoors NOW. Avoid open fields, trees, metal.",
            "hi": "तुरंत घर के अंदर जाएं। खुले मैदान, पेड़, धातु से दूर रहें।",
            "mr": "लगेच घरात जा। उघडी शेते, झाडे, धातू टाळा।",
        },
    },
    "downburst": {
        "word": {"en": "SEVERE WIND", "hi": "तेज़ आंधी", "mr": "सोसाट्याचा वारा"},
        "action": {
            "en": "Damaging gusts expected. Secure loose objects, avoid travel.",
            "hi": "तेज़ झोंके संभव। सामान सुरक्षित करें, यात्रा टालें।",
            "mr": "जोरदार झोत अपेक्षित। सामान आवरा, प्रवास टाळा।",
        },
    },
    "cloudburst": {
        "word": {"en": "CLOUDBURST RISK", "hi": "मूसलाधार वर्षा खतरा",
                 "mr": "ढगफुटी धोका"},
        "action": {
            "en": "Intense rain likely. Avoid low areas, streams, underpasses.",
            "hi": "भारी वर्षा संभव। निचले इलाके, नाले, अंडरपास से दूर रहें।",
            "mr": "मुसळधार पाऊस संभव। सखल भाग, नाले, भुयारी मार्ग टाळा।",
        },
    },
}


def render(hazard: str, lang: str, poi: str, prob: float,
           eta_min: float | None, extra: str = "") -> str:
    """Render a short alert text (SMS-safe length for en; compact else)."""
    t = _T.get(hazard, _T["hail"])
    word = t["word"].get(lang, t["word"]["en"])
    action = t["action"].get(lang, t["action"]["en"])
    if lang == "en":
        eta = f" ETA ~{eta_min:.0f} min." if eta_min is not None else ""
        return (f"VAJRA {word} ALERT {poi}: P={prob:.0%}.{eta} "
                f"{action} (demo)")
    eta = f" ~{eta_min:.0f} min." if eta_min is not None else ""
    return f"VAJRA {word} {poi}: {prob:.0%}.{eta} {action} (demo)"
