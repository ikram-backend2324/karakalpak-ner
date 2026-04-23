# ml/rule_ner.py
# Pro-level Rule-based NER for Karakalpak numerical expressions
# Dictionaries corrected to official Karakalpak orthography (April 2026)
#
# Canonical apostrophe: straight ' (U+0027) — all variants normalized before matching
# Official source: answers provided by native Karakalpak speaker/linguist
#
# Pipeline:
#   normalize_text()   — apostrophe unification, char variants, word variants, compound splitting
#   extract()          — multi-pass pattern matching with confidence scoring
#   validate_span()    — reject impossible values (month=13, hour=25, percent=999)
#   resolve_overlaps() — highest-confidence / longest match wins
#   normalize_*()      — raw span → structured value
#   post_process()     — deduplicate, re-index, filter weak APX

from __future__ import annotations
import re, math, logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# 1.  OFFICIAL KARAKALPAK DICTIONARIES
# =============================================================================

# ── 1-10 ──────────────────────────────────────────────────────────────────────
UNITS: dict = {
    "bir":    1,
    "eki":    2,
    "u'sh":   3,   # officially u'sh  (not üsh)
    "to'rt":  4,   # officially to'rt (not tört)
    "bes":    5,
    "alti":   6,   # officially alti  (not altı)
    "jeti":   7,
    "segiz":  8,
    "tog'iz": 9,   # officially tog'iz (not toğız)
    "on":     10,
}

# ── Tens 20-90 ────────────────────────────────────────────────────────────────
TENS: dict = {
    "jigirma": 20,
    "otiz":    30,
    "qiriq":   40,   # officially qiriq  (not qırq)
    "eliw":    50,   # officially eliw   (not ellik — that is Uzbek)
    "alpis":   60,   # officially alpis  (not altmış — that is Uzbek/Old Turkic)
    "jetpis":  70,
    "seksen":  80,
    "toqsan":  90,
}

# Merge for words_to_number lookups
UNITS_ALL: dict = {**UNITS, **TENS}

# ── Multipliers ───────────────────────────────────────────────────────────────
MULTIPLIERS: dict = {
    "ju'z":    100,        # officially ju'z  (not jüz)
    "min'g":   1_000,      # officially min'g (not müng)
    "million":  1_000_000,
    "milyard":  1_000_000_000,
}

SPECIAL_MULTIPLIERS: dict = {
    "yarim":  0.5,    # officially yarim  (not yarım)
    "sherek": 0.25,   # officially sherek (not çerek)
}

# ── Currency ──────────────────────────────────────────────────────────────────
CURRENCY_WORDS: list = [
    "so'm", "tenge", "dollar", "rubl", "UZS", "KZT", "USD", "RUB",
]

CURRENCY_CANONICAL: dict = {
    "so'm": "UZS", "tenge": "KZT", "dollar": "USD", "rubl": "RUB",
    "uzs":  "UZS", "kzt":   "KZT", "usd":    "USD", "rub":  "RUB",
}

# ── Percent ───────────────────────────────────────────────────────────────────
PERCENT_WORDS: list = ["procent", "u'leske", "%"]
# Note: "procent" is the Karakalpak spelling; "protsent" accepted as variant

# ── Ordinals ──────────────────────────────────────────────────────────────────
ORDINAL_MAP: dict = {
    "birinshi":    1,
    "ekinshi":     2,
    "u'shinshi":   3,    # officially u'shinshi
    "to'rtinshi":  4,    # officially to'rtinshi
    "beshinshi":   5,
    "altinshi":    6,    # officially altinshi
    "jetinshi":    7,
    "segizinshi":  8,
    "tog'izinshi": 9,    # officially tog'izinshi
    "oninshi":     10,
}

# ── Month names ───────────────────────────────────────────────────────────────
MONTH_MAP: dict = {
    # International names used in official Karakalpak documents
    "yanvar":    1,
    "fevral":    2,
    "mart":      3,
    "aprel":     4,
    "may":       5,
    "iyun":      6,
    "iyul":      7,
    "avgust":    8,
    "sentyabr":  9,    # officially sentyabr (not sentabr/sentabır)
    "oktyabr":   10,   # officially oktyabr  (not oktabr)
    "noyabr":    11,
    "dekabr":    12,
    # Traditional Karakalpak names
    "nawriz":    3,    # Nawrız = March (spring festival month)
    "nawrız":    3,    # variant with special ı
}

# ── Weekday names (used as DAT confidence signals) ────────────────────────────
WEEKDAYS: dict = {
    "du'ysenbi":  1,   # Monday
    "seysenbi":   2,   # Tuesday
    "sa'rsenbi":  3,   # Wednesday
    "beysenbi":   4,   # Thursday
    "juma":       5,   # Friday
    "senbi":      6,   # Saturday
    "yekshenbi":  7,   # Sunday
}

# ── Year markers ─────────────────────────────────────────────────────────────
# All three grammatical case forms recognized
YEAR_MARKERS: list = [
    "jili",    # straight-apostrophe normalized form of jılı (nominative)
    "jilda",   # locative — "in 2024"
    "jil",     # shorthand / list headers
    # Keep legacy forms for backwards compatibility
    "jılı",
    "jılda",
    "jıl",
]

# ── Time context ──────────────────────────────────────────────────────────────
TIME_CONTEXT: dict = {
    "tu'ste":  "12:00-14:00",   # midday/noon    (officially tu'ste)
    "keshte":  "18:00-21:00",   # evening
    "erten'":  "06:00-10:00",   # morning        (officially erten' with nasal n')
    "azanda":  "06:00-10:00",   # morning (literary/speech variant)
    "tu'nde":  "22:00-06:00",   # night          (officially tu'nde)
}

# ── Fraction keywords ─────────────────────────────────────────────────────────
# Both European comma style and scientific dot style supported
FRACTION_KEYWORDS: dict = {
    "bu'tin":  "whole part separator",   # officially bu'tin (not butın)
    "nu'kte":  "decimal point",          # scientific/international style
    "u'tir":   "decimal comma",          # European/Russian style — more common in gov docs
}

# ── Approximate ──────────────────────────────────────────────────────────────
APPROX_MAP: dict = {
    "birneshe":    "~3-5",
    "bir neshe":   "~3-5",
    "birqansha":   "~5-10",
    "bir qansha":  "~5-10",
    "az":          "az (a little)",
    "ko'p":        "ko'p (many)",
    "kop":         "ko'p (many)",          # without apostrophe
    "ju'da' ko'p": "ju'da' ko'p (very many)",
    "juda ko'p":   "ju'da' ko'p (very many)",
    "juda kop":    "ju'da' ko'p (very many)",  # without apostrophes
    "tolip":       "ko'p (many)",
}

# ── Measurement units ────────────────────────────────────────────────────────
UNIT_WORDS: set = {
    "gektar",  # hectare
    "adam",    # person
    "kisi",    # person (variant)
    "dana",    # piece / unit
    "bas",     # head (livestock)  — officially bas (not baş)
    "tonna",   # ton
    "litr",    # litre
    "jil",     # year              — officially jil (not yıl)
    "ay",      # month
    "ku'n",    # day               — officially ku'n (not kün)
    "minut",   # minute
    "sekund",  # second
    "metr",    # metre
    "km",      # kilometre
    "sm",      # centimetre
    "m2",      # square metre
    "km2",     # square kilometre
}

# ── Abbreviations ────────────────────────────────────────────────────────────
ABBREVIATIONS: dict = {
    "mln":  "million",
    "mlrd": "milyard",
    "trln": "trilyon",
    "s.":   "saat",    # time abbreviation (10-s. = 10 o'clock)
    "min.": "minut",
}

# Context words — BIR disambiguation
_BIR_NONNUMERIC = {"neshe", "qansha", "az", "ko'p", "marta", "ret", "gezek"}
_BIR_NUMERIC    = set(MULTIPLIERS) | set(UNITS_ALL) | set(CURRENCY_WORDS) | set(PERCENT_WORDS)


# =============================================================================
# 2.  RESULT DATACLASS
# =============================================================================

@dataclass
class NERResult:
    id:          int
    type:        str
    raw:         str            # original user input span
    formatted:   str            # normalized human-readable value
    value:       object         # numeric float or string
    unit:        str
    confidence:  float          # 0.0 – 1.0
    signals:     list
    sent_idx:    int
    start:       int
    end:         int
    debug_trace: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "type":        self.type,
            "raw":         self.raw,
            "formatted":   self.formatted,
            "value":       str(self.value),
            "unit":        self.unit,
            "confidence":  round(self.confidence, 3),
            "signals":     self.signals,
            "sent_idx":    self.sent_idx,
            "start":       self.start,
            "end":         self.end,
        }


# =============================================================================
# 3.  NORMALIZATION PIPELINE
# =============================================================================

# ── 3a. Apostrophe unification ────────────────────────────────────────────────
# ALL apostrophe-like characters → straight ' (U+0027)
# This must run FIRST before any dictionary lookups
_APOSTROPHE_VARIANTS = re.compile(
    r"[\u2018\u2019\u201a\u201b\u02bc\u02b9\u0060\u00b4\u02c8\u02ca\u02cb`´ʻʼ']"
)

def _unify_apostrophes(text: str) -> str:
    """Convert every apostrophe variant to straight ' (U+0027)."""
    return _APOSTROPHE_VARIANTS.sub("'", text)


# ── 3b. Character-level transliteration ──────────────────────────────────────

def _normalize_chars(text: str) -> str:
    """
    Cyrillic lookalikes → Latin, special Unicode letters → canonical Latin.
    Apostrophes are already unified by _unify_apostrophes() before this runs.
    """
    _SINGLE = {
        # Cyrillic letters that look identical to Latin
        "а": "a", "е": "e", "о": "o", "р": "r", "с": "s",
        "у": "u", "х": "x", "к": "k", "н": "n", "т": "t",
        "л": "l", "и": "i", "п": "p", "м": "m", "в": "v",
        "А": "A", "Е": "E", "О": "O", "Р": "R", "С": "S",
        "У": "U", "Х": "X", "К": "K", "Н": "N", "Т": "T",
        # Karakalpak/Uzbek special letters that may appear
        "ğ": "g'",  # ğ → g'   (then word map handles tog'iz etc.)
        "ü": "u'",  # ü → u'
        "ö": "o'",  # ö → o'
        "ş": "sh",  # ş → sh   (word map handles u'sh etc.)
        "ı": "i",   # ı → i    (word map handles jili etc.)
        "ğ": "g'",
        "ñ": "n'",  # ñ → n'   (word map handles erten' etc.)
        "ŋ": "n'",
        "ң": "n'",
        "ғ": "g'",
        "Ғ": "G'",
        "ү": "u'",
        "Ү": "U'",
        "і": "i",
    }
    for src, dst in _SINGLE.items():
        text = text.replace(src, dst)
    return text


# ── 3c. Abbreviation expansion ───────────────────────────────────────────────

def _expand_abbreviations(text: str) -> str:
    """Expand mln → million, mlrd → milyard, etc."""
    for abbr, full in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)
    return text


# ── 3d. Word-level variant map ────────────────────────────────────────────────
# Maps every known misspelling / variant → canonical Karakalpak spelling
# Keys are lowercase; values are canonical lowercase

_WORD_VARIANTS: dict = {
    # ── 1-10 ─────────────────────────────────────────────────────────────────
    # 3 — u'sh
    "ush":   "u'sh", "üsh": "u'sh", "üch": "u'sh", "uch": "u'sh",
    "uç":    "u'sh", "üç":  "u'sh",
    # 4 — to'rt
    "tort":  "to'rt", "tört": "to'rt", "dört": "to'rt",
    # 5 — bes
    "besh":  "bes",   "beş":  "bes",
    # 6 — alti
    "altı":  "alti",  "olti": "alti",
    # 9 — tog'iz
    "togiz":   "tog'iz", "toğız": "tog'iz", "togız": "tog'iz",
    "toğiz":   "tog'iz", "toquz": "tog'iz", "toxiz": "tog'iz",
    # ── Tens ─────────────────────────────────────────────────────────────────
    # 40 — qiriq
    "qırq":  "qiriq", "qirq": "qiriq", "kirk": "qiriq", "kırk": "qiriq",
    # 50 — eliw  (ellik is Uzbek — accepted as variant only)
    "ellik": "eliw",  "ellic": "eliw",
    # 60 — alpis  (altmış/altmish is Uzbek/Old Turkic)
    "altmış": "alpis", "altmish": "alpis", "oltmish": "alpis",
    "altmis":  "alpis",
    # 70 — jetpis
    "yetmish": "jetpis", "yetpis": "jetpis",
    # 80 — seksen
    "seksan":  "seksen",
    # 90 — toqsan
    "toxsan":  "toqsan", "toqusan": "toqsan", "toqson": "toqsan",
    # ── Multipliers ───────────────────────────────────────────────────────────
    # 100 — ju'z
    "juz":  "ju'z", "yuz":  "ju'z", "jüz": "ju'z",
    # 1000 — min'g
    "ming": "min'g", "müng": "min'g", "mung": "min'g", "miñ": "min'g",
    # million
    "milion": "million", "millyon": "million",
    # milyard
    "milliard": "milyard", "miliard": "milyard",
    # special multipliers
    "yarim":  "yarim",  "yarım": "yarim",
    "sherek": "sherek", "çerek": "sherek", "cherek": "sherek",
    # ── Ordinals ──────────────────────────────────────────────────────────────
    "birinchi":     "birinshi",
    "birinci":      "birinshi",
    "ikkinchi":     "ekinshi",
    "ikinci":       "ekinshi",
    "ekinchi":      "ekinshi",
    "uchinchi":     "u'shinshi",
    "üshinshi":     "u'shinshi",
    "ushinshi":     "u'shinshi",
    "üshinchi":     "u'shinshi",
    "tortinchi":    "to'rtinshi",
    "törtinshi":    "to'rtinshi",
    "törtinchi":    "to'rtinshi",
    "beshinchi":    "beshinshi",
    "oltinchi":     "altinshi",
    "altınshı":     "altinshi",
    "yettinchi":    "jetinshi",
    "sakkizinchi":  "segizinshi",
    "sekkizinchi":  "segizinshi",
    "togizinchi":   "tog'izinshi",
    "toqqizinchi":  "tog'izinshi",
    "togizinshi":   "tog'izinshi",   # without apostrophe
    "tortinshi":    "to'rtinshi",    # without apostrophe
    "onınchi":      "oninshi",
    "onunchi":      "oninshi",
    # ── Currency ──────────────────────────────────────────────────────────────
    "som":   "so'm",
    "sum":   "so'm",
    "сум":   "so'm",
    "сўм":   "so'm",
    # ── Percent ───────────────────────────────────────────────────────────────
    "protsent": "procent",   # Uzbek/Russian spelling → Karakalpak
    "protcent": "procent",
    "pratsent": "procent",
    "prosent":  "procent",
    "persent":  "procent",
    "foiz":     "procent",   # Uzbek word
    "uleske":   "u'leske",
    "üleske":   "u'leske",
    # ── Time context ──────────────────────────────────────────────────────────
    "tuste":    "tu'ste",
    "tüste":    "tu'ste",
    "tunde":    "tu'nde",
    "tünde":    "tu'nde",
    "erten":    "erten'",
    "ertan":    "erten'",
    "erteng":   "erten'",   # old variant used in our previous version
    "ertalab":  "erten'",   # Uzbek influence
    "kechte":   "keshte",   # Uzbek influence
    "kechqurun":"keshte",
    # ── Fraction keywords ─────────────────────────────────────────────────────
    "butin":  "bu'tin",
    "butın":  "bu'tin",
    "butun":  "bu'tin",
    "nuqta":  "nu'kte",    # old spelling in our previous version
    "nukta":  "nu'kte",
    "nukte":  "nu'kte",    # without apostrophe
    "utir":   "u'tir",
    # ── Approximate ───────────────────────────────────────────────────────────
    "birneshe":  "birneshe",  # already canonical
    "bir neche": "bir neshe",
    "birneche":  "bir neshe",
    "birqancha": "birqansha",
    "bir qancha":"bir qansha",
    "birkancha": "birqansha",
    "köp":       "ko'p",
    "öte köp":   "ju'da' ko'p",
    "juda ko'p": "ju'da' ko'p",
    "juda kop":  "ju'da' ko'p",    # without apostrophes
    "judakop":   "ju'da' ko'p",    # compound form
    # ── Units ─────────────────────────────────────────────────────────────────
    "baş":  "bas",
    "bosh": "bas",
    "kisi": "kisi",  # already canonical
    "kün":  "ku'n",
    "kun":  "ku'n",
    "yil":  "jil",   # Uzbek
    "yıl":  "jil",   # old variant
    "jılı": "jili",  # year marker variants → normalized form
    "jılda":"jilda",
    "jıl":  "jil",
    # ── Month name corrections ─────────────────────────────────────────────────
    "sentabr":   "sentyabr",
    "sentabır":  "sentyabr",
    "oktyabır":  "oktyabr",
    "oktabr":    "oktyabr",
    # ── Weekdays ──────────────────────────────────────────────────────────────
    "duysenbi":  "du'ysenbi",
    "düysenbi":  "du'ysenbi",
}


def _normalize_word(word: str) -> str:
    """Apply word-level variant map. Preserves leading capital."""
    low = word.lower()
    if low in _WORD_VARIANTS:
        canonical = _WORD_VARIANTS[low]
        return canonical[0].upper() + canonical[1:] if word and word[0].isupper() else canonical
    return word


# ── 3e. Compound splitter ─────────────────────────────────────────────────────
# Handles "onbir" → "on bir", "jigirmabesmingsom" → "jigirma bes min'g so'm"
# Includes ALL variant spellings so compound forms of variants also split

_ALL_KNOWN_WORDS = sorted(
    list(UNITS_ALL) + list(MULTIPLIERS) + list(SPECIAL_MULTIPLIERS) +
    list(ORDINAL_MAP) + list(MONTH_MAP) + list(WEEKDAYS) +
    CURRENCY_WORDS + PERCENT_WORDS + list(UNIT_WORDS) +
    ["saat", "bu'tin", "nu'kte", "u'tir", "min'g", "ju'z",
     "million", "milyard", "jili", "jilda", "jil",
     "birneshe", "birqansha"] +
    list(_WORD_VARIANTS.keys()),
    key=len, reverse=True
)

_COMPOUND_RE = re.compile(
    r"^(" + "|".join(re.escape(w) for w in _ALL_KNOWN_WORDS) + r")",
    re.IGNORECASE | re.UNICODE
)


def _split_compound(token: str) -> str:
    """'jigirmabir' → 'jigirma bir', 'onbir' → 'on bir'"""
    tl = token.lower()
    if tl in {w.lower() for w in _ALL_KNOWN_WORDS}:
        return token
    remaining, parts = tl, []
    while remaining:
        m = _COMPOUND_RE.match(remaining)
        if m:
            parts.append(m.group(1))
            remaining = remaining[m.end():]
        else:
            return token   # can't fully decompose — return unchanged
    return " ".join(parts) if not remaining else token


# ── 3f. Full normalize_text pipeline ─────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Six-pass pipeline before pattern matching:
      1. Apostrophe unification (all variants → straight ')
      2. Character-level transliteration (Cyrillic lookalikes → Latin)
      3. Abbreviation expansion (mln → million)
      4. Word-level variant normalization (juz → ju'z, ellik → eliw …)
      5. Compound word splitting (onbir → on bir)
      6. Word-level normalization again on freshly split tokens
    """
    def _word_pass(t: str) -> str:
        return "".join(
            tok if re.match(r"\s+", tok) else _normalize_word(tok)
            for tok in re.split(r"(\s+)", t)
        )

    def _split_pass(t: str) -> str:
        return "".join(
            tok if re.match(r"\s+", tok) else _split_compound(tok)
            for tok in re.split(r"(\s+)", t)
        )

    text = _unify_apostrophes(text)      # 1
    text = _normalize_chars(text)        # 2
    text = _expand_abbreviations(text)   # 3
    text = _word_pass(text)              # 4
    text = _split_pass(text)             # 5
    text = _word_pass(text)              # 6 — normalize tokens produced by split
    return text


# =============================================================================
# 4.  REGEX PATTERNS
# =============================================================================

_all_num = list(UNITS_ALL) + list(MULTIPLIERS) + list(SPECIAL_MULTIPLIERS)
_NUM_PAT    = "|".join(sorted(_all_num,          key=len, reverse=True))
_MONTH_PAT  = "|".join(sorted(MONTH_MAP,         key=len, reverse=True))
_ORD_PAT    = "|".join(sorted(ORDINAL_MAP,       key=len, reverse=True))
_CURR_PAT   = "|".join(re.escape(c) for c in sorted(CURRENCY_WORDS,  key=len, reverse=True))
_PCT_PAT    = "|".join(re.escape(p) for p in sorted(PERCENT_WORDS,   key=len, reverse=True))
_APPROX_PAT = "|".join(re.escape(a) for a in sorted(APPROX_MAP,      key=len, reverse=True))
_TIM_PAT    = "|".join(re.escape(t) for t in sorted(TIME_CONTEXT,    key=len, reverse=True))
_WEEK_PAT   = "|".join(re.escape(w) for w in sorted(WEEKDAYS,        key=len, reverse=True))
_YEAR_PAT   = "|".join(re.escape(y) for y in sorted(YEAR_MARKERS,    key=len, reverse=True))
_UNIT_PAT   = "|".join(re.escape(u) for u in sorted(UNIT_WORDS,      key=len, reverse=True))
_F = re.IGNORECASE | re.UNICODE


def _build_patterns() -> dict:
    mon = (
        r"(?:(?:\d[\d\s]*(?:min'g|million|milyard)?)"
        r"|(?:(?:yarim|sherek)\s+(?:min'g|million|milyard))"
        r"|(?:(?:" + _NUM_PAT + r")\s*(?:(?:" + _NUM_PAT + r")\s*)*))"
        r"\s*(?:" + _CURR_PAT + r")"
    )
    pct = (
        r"(?:\d+(?:[.,]\d+)?\s*%"
        r"|(?:\d+|(?:" + _NUM_PAT + r")(?:\s+(?:" + _NUM_PAT + r"))*)"
        r"\s+(?:" + _PCT_PAT + r"))"
    )
    # DAT: supports jili / jilda / jil / nawrız / weekday-prefixed dates
    dat = (
        r"(?:(?:" + _WEEK_PAT + r"),?\s*)?"           # optional weekday prefix
        r"(?:"
        r"\d{4}[-\s](?:" + _YEAR_PAT + r")(?:\s+\d{1,2}[-\s](?:" + _MONTH_PAT + r"))?"
        r"|\d{1,2}[-\s.](?:" + _MONTH_PAT + r")(?:[-\s.]\d{2,4})?"
        r"|(?:" + _MONTH_PAT + r")\s+\d{1,2}(?:[-\s,]\s*\d{2,4})?"
        r"|\d{1,2}[./]\d{1,2}[./]\d{2,4}"
        r")"
    )
    tim = (
        r"(?:saat\s+(?:\d{1,2}(?::\d{2})?|(?:" + _NUM_PAT + r")(?:\s+(?:" + _NUM_PAT + r"))*)"
        r"|(?:" + _TIM_PAT + r"))"
    )
    cnt = (
        r"(?:(?:\d[\d\s,.]*)|(?:(?:" + _NUM_PAT + r")(?:\s+(?:" + _NUM_PAT + r"))*))"
        r"\s+(?:" + _UNIT_PAT + r")"
    )
    frc = (
        r"(?:\d+[.,]\d+"
        r"|(?:(?:" + _NUM_PAT + r")\s+)*bu'tin(?:\s+(?:" + _NUM_PAT + r"))+"
        r"|(?:nu'kte|u'tir)\s+(?:" + _NUM_PAT + r")(?:\s+(?:" + _NUM_PAT + r"))*)"
    )
    return {
        "MON": re.compile(mon, _F),
        "PCT": re.compile(pct, _F),
        "DAT": re.compile(dat, _F),
        "TIM": re.compile(tim, _F),
        "ORD": re.compile(r"\b(?:" + _ORD_PAT    + r")\b", _F),
        "CNT": re.compile(cnt, _F),
        "FRC": re.compile(frc, _F),
        "APX": re.compile(r"\b(?:" + _APPROX_PAT + r")\b", _F),
    }


PATTERNS = _build_patterns()
_PRIORITY = ["MON", "DAT", "TIM", "PCT", "CNT", "FRC", "ORD", "APX"]


# =============================================================================
# 5.  CONFIDENCE SCORING
# =============================================================================

_SIGNAL_WEIGHTS = {
    "has_currency_word":     0.35,
    "has_percent_word":      0.35,
    "has_year_marker":       0.35,
    "has_saat_prefix":       0.35,
    "has_ordinal_suffix":    0.35,
    "has_unit_word":         0.30,
    "has_butin_keyword":     0.30,
    "has_approx_phrase":     0.30,
    "has_weekday_prefix":    0.25,   # weekday before date → strong DAT signal
    "has_digit":             0.20,
    "has_month_name":        0.20,
    "has_num_word":          0.15,
    "has_multiplier":        0.15,
    "span_multi_word":       0.10,
    "value_in_range":        0.10,
    "misspelled_but_matched":0.00,   # neutral — just informational signal
    # penalties
    "bir_likely_article":   -0.25,
    "on_likely_prep":       -0.20,
    "value_out_of_range":   -0.40,
    "single_ambiguous":     -0.10,
}

_BASE_CONF = {
    "MON": 0.30, "PCT": 0.30, "DAT": 0.30, "TIM": 0.30,
    "CNT": 0.25, "FRC": 0.25, "ORD": 0.35, "APX": 0.40,
}


def _score_match(etype: str, raw: str, ctx_before: str, ctx_after: str,
                 original_raw: str = "") -> tuple:
    raw_l   = raw.lower()
    tokens  = raw_l.split()
    signals = []
    score   = _BASE_CONF.get(etype, 0.25)

    # Universal signals
    if re.search(r"\d", raw):                        signals.append("has_digit")
    if any(t in UNITS_ALL for t in tokens):          signals.append("has_num_word")
    if any(t in MULTIPLIERS for t in tokens):        signals.append("has_multiplier")
    if len(tokens) >= 2:                             signals.append("span_multi_word")
    else:                                            signals.append("single_ambiguous")

    # Type-specific signals
    if etype == "MON":
        if any(c.lower() in raw_l for c in CURRENCY_WORDS):
            signals.append("has_currency_word")
    elif etype == "PCT":
        if any(p in raw_l for p in PERCENT_WORDS):
            signals.append("has_percent_word")
    elif etype == "DAT":
        if any(y in raw_l for y in YEAR_MARKERS):   signals.append("has_year_marker")
        if any(m in raw_l for m in MONTH_MAP):       signals.append("has_month_name")
        if any(w in raw_l for w in WEEKDAYS):        signals.append("has_weekday_prefix")
    elif etype == "TIM":
        if raw_l.startswith("saat"):                 signals.append("has_saat_prefix")
        if any(t in raw_l for t in TIME_CONTEXT):    signals.append("has_approx_phrase")
    elif etype == "CNT":
        if any(u in tokens for u in UNIT_WORDS):     signals.append("has_unit_word")
    elif etype == "FRC":
        if "bu'tin" in raw_l:                        signals.append("has_butin_keyword")
    elif etype == "ORD":
        if any(o in raw_l for o in ORDINAL_MAP):     signals.append("has_ordinal_suffix")
    elif etype == "APX":
        if any(a in raw_l for a in APPROX_MAP):      signals.append("has_approx_phrase")

    # Misspelling signal (informational only, no penalty)
    if original_raw and original_raw.lower() != raw_l:
        signals.append("misspelled_but_matched")

    # Ambiguity penalties
    ctx_tokens = (ctx_before + " " + ctx_after).lower().split()
    if tokens == ["bir"] or (etype in ("CNT", "APX") and tokens[:1] == ["bir"]):
        next_tok = ctx_after.strip().split()[:1]
        next_tok = next_tok[0] if next_tok else ""
        if next_tok in _BIR_NONNUMERIC or not any(t in _BIR_NUMERIC for t in ctx_tokens):
            signals.append("bir_likely_article")
    if tokens == ["on"] and etype == "CNT":
        signals.append("on_likely_prep")

    for sig in signals:
        score += _SIGNAL_WEIGHTS.get(sig, 0.0)

    return round(max(0.0, min(1.0, score)), 3), signals


# =============================================================================
# 6.  VALIDATION
# =============================================================================

def _validate_span(etype: str, value, raw: str) -> tuple:
    """Return (is_valid, reason). Rejects physically impossible values."""
    try:
        if etype == "PCT":
            v = float(value)
            if v < 0:   return False, f"negative percent {v}"
            if v > 100: return False, f"percent > 100: {v}"
        elif etype == "DAT":
            parts = str(value).split("-")
            if len(parts) == 3:
                y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
                if not (1 <= mo <= 12):    return False, f"invalid month {mo}"
                if not (1 <= d  <= 31):    return False, f"invalid day {d}"
                if not (1900 <= y <= 2200): return False, f"year {y} out of range"
        elif etype == "TIM":
            v = str(value)
            if ":" in v and "-" not in v:
                h = int(v.split(":")[0])
                if not (0 <= h <= 23): return False, f"invalid hour {h}"
        elif etype in ("MON", "CNT"):
            v = float(value) if value != "" else 0
            if v < 0: return False, f"negative {etype} value {v}"
    except (ValueError, TypeError) as e:
        return False, f"parse error: {e}"
    return True, "ok"


# =============================================================================
# 7.  NORMALIZERS
# =============================================================================

def words_to_number(tokens: list) -> float:
    """Convert Karakalpak number word list to float.
    e.g. ["bes", "ju'z", "min'g"] → 500 000"""
    result = current = 0.0
    for tok in tokens:
        tl = tok.lower().strip(".,;:!?\"'")
        if re.fullmatch(r"\d+", tl):
            current += int(tl)
        elif re.fullmatch(r"\d+[.,]\d+", tl):
            current += float(tl.replace(",", "."))
        elif tl in UNITS_ALL:
            current += UNITS_ALL[tl]
        elif tl in MULTIPLIERS:
            mult = MULTIPLIERS[tl]
            if mult >= 1000:
                result  += (current or 1) * mult
                current  = 0.0
            else:
                current  = (current or 1) * mult
        elif tl in SPECIAL_MULTIPLIERS:
            current = (current or 1) * SPECIAL_MULTIPLIERS[tl]
    return result + current


def _norm_MON(raw: str) -> dict:
    tokens = raw.split()
    currency, num_toks = "UZS", []
    for tok in tokens:
        mapped = CURRENCY_CANONICAL.get(tok.lower().strip(".,"))
        if mapped: currency = mapped
        else:      num_toks.append(tok)
    value = words_to_number(num_toks)
    return {"value": value, "unit": currency, "formatted": f"{value:,.0f} {currency}"}


def _norm_PCT(raw: str) -> dict:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", raw)
    if m:
        v = float(m.group(1).replace(",", "."))
        return {"value": v, "unit": "%", "formatted": f"{v:.1f}%"}
    tokens   = raw.split()
    num_toks = [t for t in tokens if t.lower() not in ("procent", "u'leske", "%")]
    for tok in tokens:
        if tok.lower().endswith(("ten", "tan", "den", "dan", "nen", "nan")):
            base = words_to_number([tok[:-3]])
            if base:
                v = round(100 / base, 2)
                return {"value": v, "unit": "%", "formatted": f"{v:.1f}%"}
    v = words_to_number(num_toks)
    return {"value": v, "unit": "%", "formatted": f"{v:.1f}%"}


def _norm_DAT(raw: str) -> dict:
    raw_l = raw.lower().strip()
    day = month = year = None

    # Remove weekday prefix if present
    for wd in WEEKDAYS:
        raw_l = re.sub(r"^" + re.escape(wd) + r",?\s*", "", raw_l)

    # Numeric format dd.mm.yyyy
    m = re.match(r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})", raw_l)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100: year += 2000
        return {"value": f"{year}-{month:02d}-{day:02d}",
                "unit": "date", "formatted": f"{day:02d}.{month:02d}.{year}"}

    # Year marker
    for ym in sorted(YEAR_MARKERS, key=len, reverse=True):
        m = re.search(r"(\d{4})[-\s]" + re.escape(ym), raw_l)
        if m:
            year = int(m.group(1))
            break

    # Day digit
    m = re.search(r"\b(\d{1,2})\b", raw_l)
    if m:
        c = int(m.group(1))
        if c <= 31: day = c

    # Month name
    for name, num in sorted(MONTH_MAP.items(), key=lambda x: -len(x[0])):
        if name in raw_l:
            month = num
            break

    y, mo, d = year or 2024, month or 1, day or 1
    return {"value": f"{y}-{mo:02d}-{d:02d}", "unit": "date",
            "formatted": f"{d:02d}.{mo:02d}.{y}"}


def _norm_TIM(raw: str) -> dict:
    raw_l = raw.lower().strip()
    for word, rng in TIME_CONTEXT.items():
        if word in raw_l:
            return {"value": rng, "unit": "time_range", "formatted": rng}
    m = re.search(r"(\d{1,2}):(\d{2})", raw_l)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        return {"value": f"{h:02d}:{mn:02d}", "unit": "time",
                "formatted": f"{h:02d}:{mn:02d}"}
    num_toks = [t for t in raw_l.split() if t != "saat"]
    hour = int(words_to_number(num_toks)) if num_toks else 0
    return {"value": f"{hour:02d}:00", "unit": "time", "formatted": f"{hour:02d}:00"}


def _norm_CNT(raw: str) -> dict:
    tokens = raw.split()
    unit, num_toks = "", []
    for tok in tokens:
        if tok.lower() in UNIT_WORDS: unit = tok
        else: num_toks.append(tok)
    value = words_to_number(num_toks)
    return {"value": value, "unit": unit, "formatted": f"{value:,.0f} {unit}".strip()}


def _norm_FRC(raw: str) -> dict:
    raw_l = raw.lower().strip()
    # Plain decimal: 1.5 or 1,5
    m = re.match(r"^(\d+)[.,](\d+)$", raw_l)
    if m:
        v = float(raw_l.replace(",", "."))
        return {"value": v, "unit": "fraction", "formatted": str(v)}
    # "bir bu'tin bes" pattern
    if "bu'tin" in raw_l:
        parts    = raw_l.split("bu'tin")
        whole    = words_to_number(parts[0].split())
        frac_raw = parts[1].strip() if len(parts) > 1 else ""
        frac_v   = words_to_number(frac_raw.split()) if frac_raw else 0.0
        if frac_v >= 1:
            digits = math.floor(math.log10(frac_v)) + 1
            frac_v /= (10 ** digits)
        v = whole + frac_v
        return {"value": round(v, 6), "unit": "fraction", "formatted": str(round(v, 4))}
    # "nu'kte / u'tir toqsan bes" pattern
    for kw in ("nu'kte", "u'tir"):
        if kw in raw_l:
            num_toks = raw_l.replace(kw, "").split()
            frac_v   = words_to_number(num_toks)
            if frac_v >= 1:
                digits = math.floor(math.log10(frac_v)) + 1
                frac_v /= (10 ** digits)
            return {"value": round(frac_v, 6), "unit": "fraction",
                    "formatted": str(round(frac_v, 4))}
    v = words_to_number(raw_l.split())
    return {"value": v, "unit": "fraction", "formatted": str(v)}


def _norm_ORD(raw: str) -> dict:
    raw_l = raw.lower().strip()
    for word, num in ORDINAL_MAP.items():
        if word in raw_l:
            return {"value": num, "unit": "ordinal", "formatted": f"{num}-shi orın"}
    clean = re.sub(r"(inshi|izinshi|inchi|shi)$", "", raw_l).strip()
    num   = int(words_to_number(clean.split())) if clean else 0
    return {"value": num, "unit": "ordinal", "formatted": f"{num}-shi orın"}


def _norm_APX(raw: str) -> dict:
    raw_l = raw.lower().strip()
    for phrase, meaning in sorted(APPROX_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in raw_l:
            return {"value": meaning, "unit": "approx", "formatted": meaning}
    return {"value": "~?", "unit": "approx", "formatted": raw}


_NORMALIZERS = {
    "MON": _norm_MON, "PCT": _norm_PCT, "DAT": _norm_DAT,
    "TIM": _norm_TIM, "CNT": _norm_CNT, "FRC": _norm_FRC,
    "ORD": _norm_ORD, "APX": _norm_APX,
}


# =============================================================================
# 8.  OVERLAP RESOLUTION
# =============================================================================

def _resolve_overlaps(matches: list) -> list:
    """Highest confidence wins; ties go to longer span, then earlier priority."""
    matches.sort(key=lambda x: (
        x["start"],
        -(x["end"] - x["start"]),
        -x.get("confidence", 0),
        _PRIORITY.index(x["type"]) if x["type"] in _PRIORITY else 99,
    ))
    resolved, last_end = [], -1
    for m in matches:
        if m["start"] >= last_end:
            resolved.append(m)
            last_end = m["end"]
        elif resolved and m.get("confidence", 0) > resolved[-1].get("confidence", 0) + 0.15:
            resolved[-1] = m
            last_end = m["end"]
    return resolved


# =============================================================================
# 9.  SENTENCE / OFFSET HELPERS
# =============================================================================

def _build_sent_offsets(text: str) -> list:
    sentences, offsets, pos = re.split(r"(?<=[.!?])\s+", text), [], 0
    for i, s in enumerate(sentences):
        offsets.append((pos, pos + len(s), i))
        pos += len(s) + 1
    return offsets


def _get_sent_idx(start: int, offsets: list) -> int:
    for s0, s1, idx in offsets:
        if s0 <= start < s1:
            return idx
    return 0


# =============================================================================
# 10. POST-PROCESSING
# =============================================================================

def _post_process(results: list) -> list:
    """Remove exact duplicates, filter weak single-word APX, re-index."""
    seen, final = set(), []
    for r in sorted(results, key=lambda x: (x.sent_idx, x.start)):
        key = (r.type, r.raw.lower(), r.sent_idx)
        if key in seen:
            r.debug_trace.append("removed_duplicate")
            continue
        seen.add(key)
        if r.type == "APX" and r.confidence < 0.35 and len(r.raw.split()) == 1:
            r.debug_trace.append(f"filtered_weak_apx:{r.confidence:.2f}")
            continue
        final.append(r)
    for i, r in enumerate(final, 1):
        r.id = i
    return final


# =============================================================================
# 11. MAIN EXTRACT
# =============================================================================

def extract(text: str, debug: bool = False) -> list:
    """
    Full extraction pipeline.

    Args:
        text:  Raw input — any apostrophe variant, any script
        debug: If True, populate debug_trace and emit log lines

    Returns:
        List of NERResult sorted by (sent_idx, start).
    """
    normalized   = normalize_text(text)
    sent_offsets = _build_sent_offsets(normalized)
    if debug:
        logger.debug("Normalized: %r", normalized)

    raw_matches = []
    for etype in _PRIORITY:
        for m in PATTERNS[etype].finditer(normalized):
            span = m.group().strip()
            if not span:
                continue
            start, end = m.start(), m.end()
            ctx_b = normalized[max(0, start - 40):start]
            ctx_a = normalized[end:end + 40]
            conf, sigs = _score_match(etype, span, ctx_b, ctx_a)
            raw_matches.append({
                "type": etype, "raw": span, "start": start, "end": end,
                "confidence": conf, "signals": sigs,
            })

    resolved = _resolve_overlaps(raw_matches)

    results = []
    for idx, match in enumerate(resolved):
        etype, raw, trace = match["type"], match["raw"], []
        if debug:
            trace += [f"pattern:{etype}", f"signals:{','.join(match['signals'])}"]

        normalizer = _NORMALIZERS.get(etype)
        try:
            norm = normalizer(raw) if normalizer else {}
        except Exception as exc:
            norm = {"value": raw, "unit": "", "formatted": raw}
            trace.append(f"normalizer_error:{exc}")
            logger.warning("Normalizer error %s %r: %s", etype, raw, exc)

        value     = norm.get("value", "")
        unit      = norm.get("unit", "")
        formatted = norm.get("formatted", raw)

        valid, reason = _validate_span(etype, value, raw)
        if not valid:
            trace.append(f"validation_failed:{reason}")
            match["signals"].append("value_out_of_range")
            match["confidence"] = max(0.0, match["confidence"] + _SIGNAL_WEIGHTS["value_out_of_range"])
        elif debug:
            trace.append(f"valid:{reason}")

        results.append(NERResult(
            id=idx + 1, type=etype, raw=raw, formatted=formatted,
            value=value, unit=unit,
            confidence=match["confidence"], signals=match["signals"],
            sent_idx=_get_sent_idx(match["start"], sent_offsets),
            start=match["start"], end=match["end"],
            debug_trace=trace,
        ))

    return _post_process(results)


# =============================================================================
# 12. PUBLIC API
# =============================================================================

def extract_from_sentences(text: str, debug: bool = False) -> tuple:
    """
    Main function used by Django views.
    Returns (results_as_dicts, sentences, token_count).
    """
    sentences   = re.split(r"(?<=[.!?])\s+", text.strip())
    ner_results = extract(text, debug=debug)
    token_count = len(text.split())
    return [r.to_dict() for r in ner_results], sentences, token_count


def extract_rich(text: str, debug: bool = False) -> tuple:
    """Returns NERResult objects instead of dicts."""
    sentences   = re.split(r"(?<=[.!?])\s+", text.strip())
    ner_results = extract(text, debug=debug)
    return ner_results, sentences, len(text.split())


def stats(results: list) -> dict:
    """Summary statistics over a list of result dicts."""
    if not results:
        return {"total": 0, "by_type": {}, "avg_confidence": 0.0}
    by_type, confidences = {}, []
    for r in results:
        t = r.get("type", "UNK")
        by_type[t] = by_type.get(t, 0) + 1
        confidences.append(float(r.get("confidence", 0)))
    avg = sum(confidences) / len(confidences)
    return {
        "total":           len(results),
        "by_type":         by_type,
        "avg_confidence":  round(avg, 3),
        "high_confidence": sum(1 for c in confidences if c >= 0.7),
        "low_confidence":  sum(1 for c in confidences if c < 0.5),
    }
