# ml/rule_ner.py
# Pro-level Rule-based NER for Karakalpak numerical expressions
#
# Pipeline:
#   normalize_text()     — char variants, word variants, compound splitting
#   extract()            — multi-pass pattern matching
#   score_match()        — confidence scoring per match
#   disambiguate()       — context-aware ambiguity resolution
#   validate_span()      — reject impossible values
#   resolve_overlaps()   — longest/highest-confidence match wins
#   normalize_*()        — convert raw span to structured value
#   post_process()       — merge, deduplicate, sort
#
# Every result carries: type, raw, formatted, value, unit,
#                       confidence, signals, sent_idx, debug_trace

from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# 1. DICTIONARIES
# =============================================================================

UNITS: dict = {
    'bir': 1, 'eki': 2, 'üsh': 3, 'tört': 4, 'bes': 5,
    'altı': 6, 'jeti': 7, 'segiz': 8, 'toğız': 9, 'on': 10,
    'jigirma': 20, 'otiz': 30, 'qırq': 40, 'ellik': 50,
    'altmış': 60, 'jetpis': 70, 'seksen': 80, 'toqsan': 90,
}

MULTIPLIERS: dict = {
    'jüz':     100,
    'müng':    1_000,
    'million': 1_000_000,
    'milyard': 1_000_000_000,
}

SPECIAL_MULTIPLIERS: dict = {
    'yarım': 0.5,
    'çerek': 0.25,
}

CURRENCY_WORDS = [
    "so'm", "tenge", "dollar", "rubl", "UZS", "KZT", "USD", "RUB",
]

CURRENCY_CANONICAL = {
    "so'm": 'UZS', 'tenge': 'KZT', 'dollar': 'USD', 'rubl': 'RUB',
    'uzs': 'UZS', 'kzt': 'KZT', 'usd': 'USD', 'rub': 'RUB',
}

PERCENT_WORDS = ['protsent', 'üleske', '%']

ORDINAL_MAP: dict = {
    'birinshi': 1, 'birinchi': 1,
    'ekinshi':  2, 'ikkinchi': 2, 'ikinci': 2,
    'üshinshi': 3, 'üshinchi': 3, 'uchinchi': 3,
    'tötinshi': 4, 'törtinshi': 4,
    'beshinshi': 5, 'beshinchi': 5,
    'altınshı': 6, 'altinchi': 6,
    'jetinshi': 7, 'yettinchi': 7,
    'segizinshi': 8, 'sakkizinchi': 8,
    'toğızınshı': 9, 'toqqizinchi': 9,
    "oninshi": 10, "o'ninchi": 10,
}

MONTH_MAP: dict = {
    'yanvar': 1,  'fevral': 2,  'mart': 3,     'aprel': 4,
    'may': 5,     'iyun': 6,    'iyul': 7,      'avgust': 8,
    'sentabr': 9, 'sentabır': 9,'oktabr': 10,   'noyabr': 11, 'dekabr': 12,
    'qantar': 1,  'aqpan': 2,   'naurız': 3,    'kokirek': 4,
    'mamır': 5,   'mavsım': 6,  'shilde': 7,    'tamız': 8,
    'qırküyrek': 9, 'qazan': 10,'noabır': 11,   'jeltoqsan': 12,
}

APPROX_MAP: dict = {
    'bir neshe': '~3–5',
    'birneshe':  '~3–5',
    'birqansha': '~5–10',
    'bir qansha':'~5–10',
    'az':        'az miqdor (a little)',
    'köp':       "ko'p (many)",
    'öte köp':   "juda ko'p (very many)",
    'tolıp':     "ko'p (many)",
}

TIME_CONTEXT: dict = {
    'tüste':  '12:00–14:00',
    'tuste':  '12:00–14:00',
    'keshte': '18:00–21:00',
    'erteng': '06:00–10:00',
    'ertalab':'06:00–10:00',
    'tünde':  '22:00–06:00',
    'tunde':  '22:00–06:00',
}

# Context words that suggest "bir" is an article, not "one"
_BIR_NONNUMERIC = {'neshe', 'qansha', 'az', 'köp', 'marta', 'ret', 'gezek'}
# Context words that confirm "bir" is numeric
_BIR_NUMERIC    = set(MULTIPLIERS) | set(UNITS) | set(CURRENCY_WORDS) | set(PERCENT_WORDS)


# =============================================================================
# 2. RESULT DATACLASS
# =============================================================================

@dataclass
class NERResult:
    id:          int
    type:        str
    raw:         str
    formatted:   str
    value:       object          # float or str
    unit:        str
    confidence:  float           # 0.0 – 1.0
    signals:     list
    sent_idx:    int
    start:       int
    end:         int
    debug_trace: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'id':          self.id,
            'type':        self.type,
            'raw':         self.raw,
            'formatted':   self.formatted,
            'value':       str(self.value),
            'unit':        self.unit,
            'confidence':  round(self.confidence, 3),
            'signals':     self.signals,
            'sent_idx':    self.sent_idx,
            'start':       self.start,
            'end':         self.end,
        }


# =============================================================================
# 3. NORMALIZATION PIPELINE
# =============================================================================

# ── 3a. Character-level ───────────────────────────────────────────────────────

def _normalize_chars(text: str) -> str:
    """
    g' → ğ, Cyrillic lookalikes → Latin.
    o' intentionally NOT converted globally (would break so'm).
    """
    text = re.sub(r"g[`ʻʼ\u2018\u2019\u201b']", 'ğ', text)
    text = re.sub(r"[`ʻʼ\u2018\u2019\u201b]", "'", text)
    _SINGLE = {
        'ġ': 'ğ', 'ғ': 'ğ', 'Ғ': 'ğ',
        'ü': 'ü', 'ű': 'ü', 'ų': 'ü', 'ү': 'ü', 'Ү': 'ü',
        'ş': 'ş', 'ш': 'ş', 'Ш': 'ş',
        'і': 'ı',
        'ö': 'ö', 'ő': 'ö',
        'ç': 'ç', 'ч': 'ç', 'Ч': 'ç',
        'ñ': 'ñ', 'ŋ': 'ñ', 'ң': 'ñ', 'Ң': 'ñ',
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'r', 'с': 's',
        'у': 'u', 'х': 'x', 'к': 'k', 'н': 'n', 'т': 't',
        'л': 'l', 'и': 'i', 'п': 'p', 'м': 'm',
        'А': 'A', 'Е': 'E', 'О': 'O', 'Р': 'R', 'С': 'S',
        'У': 'U', 'Х': 'X', 'К': 'K', 'Н': 'N', 'Т': 'T',
    }
    for src, dst in _SINGLE.items():
        text = text.replace(src, dst)
    return text


# ── 3b. Word-level variant map ────────────────────────────────────────────────

_WORD_VARIANTS: dict = {
    # 9 — toğız
    'togiz':'toğız','togız':'toğız','toğiz':'toğız',
    'toquz':'toğız','toxiz':'toğız','togis':'toğız',
    # 7 — jeti
    'yeti':'jeti','yetti':'jeti','jetti':'jeti',
    # 8 — segiz
    'sekiz':'segiz','sagiz':'segiz','sekkiz':'segiz','sakkiz':'segiz',
    # 3 — üsh
    'ush':'üsh','üch':'üsh','uch':'üsh','uç':'üsh','üç':'üsh',
    # 4 — tört
    'tort':'tört',"to'rt":'tört','dört':'tört','töt':'tört',
    # 5 — bes
    'besh':'bes','beş':'bes',
    # 6 — altı
    'alti':'altı','olti':'altı',
    # 20 — jigirma
    'yigirma':'jigirma','yigirima':'jigirma','zhigirma':'jigirma','jigirima':'jigirma',
    # 30 — otiz
    'ottiz':'otiz','utiz':'otiz','otız':'otiz',
    # 40 — qırq
    'qirq':'qırq','kirk':'qırq','kırk':'qırq','qırk':'qırq',
    # 50 — ellik
    'ellic':'ellik',
    # 60 — altmış
    'altmish':'altmış','altmis':'altmış','oltmish':'altmış','altmiş':'altmış',
    # 70 — jetpis
    'yetmish':'jetpis','yetpis':'jetpis','jetmiş':'jetpis',
    # 80 — seksen
    'seksan':'seksen',
    # 90 — toqsan
    'toxsan':'toqsan','toqusan':'toqsan','toqson':'toqsan',
    # 100 — jüz
    'juz':'jüz','yuz':'jüz',
    # 1000 — müng
    'mung':'müng','ming':'müng','miñ':'müng',
    # million / milyard
    'milion':'million','millyon':'million',
    'milliard':'milyard','miliard':'milyard',
    # special multipliers
    'yarim':'yarım','cherek':'çerek','cherak':'çerek',
    # ordinals
    'birinchi':'birinshi','birinçi':'birinshi','birinci':'birinshi',
    'ikkinchi':'ekinshi','ikinci':'ekinshi','ekinchi':'ekinshi','ikkınchi':'ekinshi',
    'uchinchi':'üshinshi','üchinchi':'üshinshi',
    'ushinshi':'üshinshi','ushinchi':'üshinshi','üshinchi':'üshinshi',
    'tortinchi':'tötinshi','törtinchi':'tötinshi',
    'beshinchi':'beshinshi','beşinchi':'beshinshi',
    'oltinchi':'altınshı','altinchi':'altınshı',
    'yettinchi':'jetinshi','ettinchi':'jetinshi',
    'sakkizinchi':'segizinshi','sekkizinchi':'segizinshi',
    'toqqizinchi':'toğızınshı','togizinchi':'toğızınshı',
    'onınchi':'oninshi','onunchi':'oninshi',
    # currency
    'som':"so'm",'sum':"so'm",'uzs':"so'm",
    # percent
    'protcent':'protsent','pratsent':'protsent','prosent':'protsent',
    'persent':'protsent','foiz':'protsent','uleske':'üleske',
    # time context
    'tuste':'tüste','tunde':'tünde','kechte':'keshte','kechqurun':'keshte',
    # fraction
    'butin':'butın','butun':'butın','nukta':'nuqta',
    # approximate
    'birneshe':'bir neshe','birneche':'bir neshe','bir neche':'bir neshe',
    'birqancha':'birqansha','bir qancha':'bir qansha','birkancha':'birqansha',
    # units
    'kishi':'kisi','bosh':'baş',
}


def _normalize_word(word: str) -> str:
    low = word.lower()
    if low in _WORD_VARIANTS:
        canonical = _WORD_VARIANTS[low]
        return canonical[0].upper() + canonical[1:] if word and word[0].isupper() else canonical
    return word


# ── 3c. Compound splitter ─────────────────────────────────────────────────────

_ALL_NUM_WORDS = sorted(
    list(UNITS) + list(MULTIPLIERS) + list(SPECIAL_MULTIPLIERS) +
    list(ORDINAL_MAP) + list(MONTH_MAP) + CURRENCY_WORDS + PERCENT_WORDS +
    ['saat','butın','nuqta','dana','gektar','kisi','adam','baş','metr',
     'tonna','litr','protsent','üleske','müng','million','milyard',
     'jılı','neshe','qansha'] +
    list(_WORD_VARIANTS.keys()),
    key=len, reverse=True
)

_COMPOUND_RE = re.compile(
    r'^(' + '|'.join(re.escape(w) for w in _ALL_NUM_WORDS) + r')',
    re.IGNORECASE | re.UNICODE
)


def _split_compound(token: str) -> str:
    """'onbir' → 'on bir', 'beshfoiz' → 'besh foiz'"""
    tl = token.lower()
    if tl in {w.lower() for w in _ALL_NUM_WORDS}:
        return token
    remaining, parts = tl, []
    while remaining:
        m = _COMPOUND_RE.match(remaining)
        if m:
            parts.append(m.group(1))
            remaining = remaining[m.end():]
        else:
            return token
    return ' '.join(parts) if not remaining else token


# ── 3d. Full pipeline ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    Four-pass pipeline:
      1. Char-level transliteration
      2. Word-level variant normalization
      3. Compound word splitting (uses variant word list)
      4. Word-level normalization again on freshly split tokens
    """
    def _word_pass(t):
        return ''.join(
            tok if re.match(r'\s+', tok) else _normalize_word(tok)
            for tok in re.split(r'(\s+)', t)
        )
    def _split_pass(t):
        return ''.join(
            tok if re.match(r'\s+', tok) else _split_compound(tok)
            for tok in re.split(r'(\s+)', t)
        )

    text = _normalize_chars(text)
    text = _word_pass(text)
    text = _split_pass(text)
    text = _word_pass(text)   # normalize tokens produced by split
    return text


# =============================================================================
# 4. REGEX PATTERNS
# =============================================================================

_canonical_num = list(UNITS) + list(MULTIPLIERS) + list(SPECIAL_MULTIPLIERS)
_NUM_WORD_PAT  = '|'.join(sorted(_canonical_num, key=len, reverse=True))
_MONTH_PAT     = '|'.join(sorted(MONTH_MAP,    key=len, reverse=True))
_ORD_PAT       = '|'.join(sorted(ORDINAL_MAP,  key=len, reverse=True))
_CURR_PAT      = '|'.join(re.escape(c) for c in sorted(CURRENCY_WORDS, key=len, reverse=True))
_PCT_PAT       = '|'.join(re.escape(p) for p in sorted(PERCENT_WORDS,  key=len, reverse=True))
_APPROX_PAT    = '|'.join(re.escape(a) for a in sorted(APPROX_MAP,     key=len, reverse=True))
_TIME_CTX_PAT  = '|'.join(re.escape(t) for t in sorted(TIME_CONTEXT,   key=len, reverse=True))
_F = re.IGNORECASE | re.UNICODE


def _build_patterns():
    mon = (
        r'(?:(?:\d[\d\s]*(?:müng|million|milyard)?)'
        r'|(?:(?:yarım|çerek)\s+(?:müng|million|milyard))'
        r'|(?:(?:' + _NUM_WORD_PAT + r')\s*(?:(?:' + _NUM_WORD_PAT + r')\s*)*))'
        r'\s*(?:' + _CURR_PAT + r')'
    )
    pct = (
        r'(?:\d+(?:[.,]\d+)?\s*%'
        r'|(?:\d+|(?:' + _NUM_WORD_PAT + r')(?:\s+(?:' + _NUM_WORD_PAT + r'))*)'
        r'\s+(?:' + _PCT_PAT + r'))'
    )
    dat = (
        r'(?:\d{4}[-\s]jılı(?:\s+\d{1,2}[-\s](?:' + _MONTH_PAT + r'))?'
        r'|\d{1,2}[-\s.](?:' + _MONTH_PAT + r')(?:[-\s.]\d{2,4})?'
        r'|(?:' + _MONTH_PAT + r')\s+\d{1,2}(?:[-\s,]\s*\d{2,4})?'
        r'|\d{1,2}[./]\d{1,2}[./]\d{2,4})'
    )
    tim = (
        r'(?:saat\s+(?:\d{1,2}(?::\d{2})?|(?:' + _NUM_WORD_PAT + r')(?:\s+(?:' + _NUM_WORD_PAT + r'))*)'
        r'|(?:' + _TIME_CTX_PAT + r'))'
    )
    cnt = (
        r'(?:(?:\d[\d\s,.]*)|(?:(?:' + _NUM_WORD_PAT + r')(?:\s+(?:' + _NUM_WORD_PAT + r'))*))'
        r'\s+(?:gektar|kisi|adam|dana|baş|metr|km|litr|tonna|sm|m2|km2|soat|minut|sekund|yıl|ay|kün)'
    )
    frc = (
        r'(?:\d+[.,]\d+'
        r'|(?:(?:' + _NUM_WORD_PAT + r')\s+)*butın(?:\s+(?:' + _NUM_WORD_PAT + r'))+'
        r'|nuqta\s+(?:' + _NUM_WORD_PAT + r')(?:\s+(?:' + _NUM_WORD_PAT + r'))*)'
    )
    return {
        'MON': re.compile(mon, _F),
        'PCT': re.compile(pct, _F),
        'DAT': re.compile(dat, _F),
        'TIM': re.compile(tim, _F),
        'ORD': re.compile(r'\b(?:' + _ORD_PAT   + r')\b', _F),
        'CNT': re.compile(cnt, _F),
        'FRC': re.compile(frc, _F),
        'APX': re.compile(r'\b(?:' + _APPROX_PAT + r')\b', _F),
    }


PATTERNS = _build_patterns()
_PRIORITY = ['MON', 'DAT', 'TIM', 'PCT', 'CNT', 'FRC', 'ORD', 'APX']


# =============================================================================
# 5. CONFIDENCE SCORING
# =============================================================================

_SIGNAL_WEIGHTS = {
    # strong signals — entity-type anchors
    'has_currency_word':    0.35,
    'has_percent_word':     0.35,
    'has_jıly_marker':      0.35,
    'has_saat_prefix':      0.35,
    'has_ordinal_suffix':   0.35,
    'has_unit_word':        0.30,
    'has_butın_keyword':    0.30,
    'has_approx_phrase':    0.30,
    # supporting
    'has_digit':            0.20,
    'has_num_word':         0.15,
    'has_multiplier':       0.15,
    'has_month_name':       0.20,
    'span_multi_word':      0.10,
    'value_in_range':       0.10,
    # penalties
    'bir_likely_article':  -0.25,
    'on_likely_prep':      -0.20,
    'value_out_of_range':  -0.40,
    'single_ambiguous':    -0.10,
}

_BASE_CONFIDENCE = {
    'MON': 0.30, 'PCT': 0.30, 'DAT': 0.30, 'TIM': 0.30,
    'CNT': 0.25, 'FRC': 0.25, 'ORD': 0.35, 'APX': 0.40,
}


def _score_match(etype: str, raw: str, ctx_before: str, ctx_after: str):
    raw_l   = raw.lower()
    tokens  = raw_l.split()
    signals = []
    score   = _BASE_CONFIDENCE.get(etype, 0.25)

    # Universal
    if re.search(r'\d', raw):         signals.append('has_digit')
    if any(t in UNITS for t in tokens): signals.append('has_num_word')
    if any(t in MULTIPLIERS for t in tokens): signals.append('has_multiplier')
    if len(tokens) >= 2:              signals.append('span_multi_word')
    else:                             signals.append('single_ambiguous')

    # Type-specific
    if etype == 'MON':
        if any(c.lower() in raw_l for c in CURRENCY_WORDS):
            signals.append('has_currency_word')
    elif etype == 'PCT':
        if any(p in raw_l for p in PERCENT_WORDS):
            signals.append('has_percent_word')
    elif etype == 'DAT':
        if 'jılı' in raw_l: signals.append('has_jıly_marker')
        if any(m in raw_l for m in MONTH_MAP): signals.append('has_month_name')
    elif etype == 'TIM':
        if raw_l.startswith('saat'):   signals.append('has_saat_prefix')
        if any(t in raw_l for t in TIME_CONTEXT): signals.append('has_approx_phrase')
    elif etype == 'CNT':
        _units = {'gektar','kisi','adam','dana','baş','metr','km','litr',
                  'tonna','sm','m2','km2','soat','minut','sekund','yıl','ay','kün'}
        if any(u in tokens for u in _units): signals.append('has_unit_word')
    elif etype == 'FRC':
        if 'butın' in raw_l: signals.append('has_butın_keyword')
    elif etype == 'ORD':
        if any(o in raw_l for o in ORDINAL_MAP): signals.append('has_ordinal_suffix')
    elif etype == 'APX':
        if any(a in raw_l for a in APPROX_MAP): signals.append('has_approx_phrase')

    # Ambiguity penalties
    ctx_tokens = (ctx_before + ' ' + ctx_after).lower().split()
    if tokens == ['bir'] or (etype in ('CNT', 'APX') and tokens[:1] == ['bir']):
        next_tok = ctx_after.strip().split()[:1]
        next_tok = next_tok[0] if next_tok else ''
        if next_tok in _BIR_NONNUMERIC:
            signals.append('bir_likely_article')
        elif not any(t in _BIR_NUMERIC for t in ctx_tokens):
            signals.append('bir_likely_article')

    if tokens == ['on'] and etype == 'CNT':
        signals.append('on_likely_prep')

    for sig in signals:
        score += _SIGNAL_WEIGHTS.get(sig, 0.0)

    return round(max(0.0, min(1.0, score)), 3), signals


# =============================================================================
# 6. VALIDATION
# =============================================================================

def _validate_span(etype: str, value, raw: str):
    """Return (is_valid, reason_string)."""
    try:
        if etype == 'PCT':
            v = float(value)
            if v < 0:      return False, f"negative percent {v}"
            if v > 100:    return False, f"percent > 100: {v}"
        elif etype == 'DAT':
            parts = str(value).split('-')
            if len(parts) == 3:
                y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
                if not (1 <= mo <= 12): return False, f"invalid month {mo}"
                if not (1 <= d  <= 31): return False, f"invalid day {d}"
                if not (1900 <= y <= 2100): return False, f"year {y} out of range"
        elif etype == 'TIM':
            v = str(value)
            if ':' in v and '–' not in v:
                h = int(v.split(':')[0])
                if not (0 <= h <= 23): return False, f"invalid hour {h}"
        elif etype in ('MON', 'CNT'):
            v = float(value) if value != '' else 0
            if v < 0: return False, f"negative {etype} {v}"
    except (ValueError, TypeError) as e:
        return False, f"parse error: {e}"
    return True, 'ok'


# =============================================================================
# 7. NORMALIZERS
# =============================================================================

def words_to_number(tokens: list) -> float:
    """['bes','jüz','müng'] → 500000.0"""
    result = current = 0.0
    for tok in tokens:
        tl = tok.lower().strip(".,;:!?\"'")
        if re.fullmatch(r'\d+', tl):
            current += int(tl)
        elif re.fullmatch(r'\d+[.,]\d+', tl):
            current += float(tl.replace(',', '.'))
        elif tl in UNITS:
            current += UNITS[tl]
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
    currency, num_toks = 'UZS', []
    for tok in tokens:
        mapped = CURRENCY_CANONICAL.get(tok.lower().strip(".,"))
        if mapped: currency = mapped
        else: num_toks.append(tok)
    value = words_to_number(num_toks)
    return {'value': value, 'unit': currency, 'formatted': f"{value:,.0f} {currency}"}


def _norm_PCT(raw: str) -> dict:
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*%', raw)
    if m:
        v = float(m.group(1).replace(',', '.'))
        return {'value': v, 'unit': '%', 'formatted': f"{v:.1f}%"}
    tokens   = raw.split()
    num_toks = [t for t in tokens if t.lower() not in ('protsent', 'üleske', '%')]
    for tok in tokens:
        if tok.lower().endswith(('ten', 'tan', 'den', 'dan')):
            base = words_to_number([tok[:-3]])
            if base:
                v = round(100 / base, 2)
                return {'value': v, 'unit': '%', 'formatted': f"{v:.1f}%"}
    v = words_to_number(num_toks)
    return {'value': v, 'unit': '%', 'formatted': f"{v:.1f}%"}


def _norm_DAT(raw: str) -> dict:
    raw_l = raw.lower().strip()
    day = month = year = None
    m = re.match(r'(\d{1,2})[./](\d{1,2})[./](\d{2,4})', raw_l)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100: year += 2000
        return {'value': f"{year}-{month:02d}-{day:02d}",
                'unit': 'date', 'formatted': f"{day:02d}.{month:02d}.{year}"}
    m = re.search(r'(\d{4})[-\s]jılı', raw_l)
    if m: year = int(m.group(1))
    m = re.search(r'\b(\d{1,2})\b', raw_l)
    if m:
        c = int(m.group(1))
        if c <= 31: day = c
    for name, num in sorted(MONTH_MAP.items(), key=lambda x: -len(x[0])):
        if name in raw_l: month = num; break
    y, mo, d = year or 2024, month or 1, day or 1
    return {'value': f"{y}-{mo:02d}-{d:02d}", 'unit': 'date',
            'formatted': f"{d:02d}.{mo:02d}.{y}"}


def _norm_TIM(raw: str) -> dict:
    raw_l = raw.lower().strip()
    for word, rng in TIME_CONTEXT.items():
        if word in raw_l:
            return {'value': rng, 'unit': 'time_range', 'formatted': rng}
    m = re.search(r'(\d{1,2}):(\d{2})', raw_l)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        return {'value': f"{h:02d}:{mn:02d}", 'unit': 'time', 'formatted': f"{h:02d}:{mn:02d}"}
    num_toks = [t for t in raw_l.split() if t != 'saat']
    hour = int(words_to_number(num_toks)) if num_toks else 0
    return {'value': f"{hour:02d}:00", 'unit': 'time', 'formatted': f"{hour:02d}:00"}


def _norm_CNT(raw: str) -> dict:
    _MEAS = {'gektar','kisi','adam','dana','baş','metr','km','litr',
             'tonna','sm','m2','km2','soat','minut','sekund','yıl','ay','kün'}
    tokens = raw.split()
    unit, num_toks = '', []
    for tok in tokens:
        if tok.lower() in _MEAS: unit = tok
        else: num_toks.append(tok)
    value = words_to_number(num_toks)
    return {'value': value, 'unit': unit, 'formatted': f"{value:,.0f} {unit}".strip()}


def _norm_FRC(raw: str) -> dict:
    raw_l = raw.lower().strip()
    m = re.match(r'^(\d+)[.,](\d+)$', raw_l)
    if m:
        v = float(raw_l.replace(',', '.'))
        return {'value': v, 'unit': 'fraction', 'formatted': str(v)}
    if 'butın' in raw_l:
        parts    = raw_l.split('butın')
        whole    = words_to_number(parts[0].split())
        frac_raw = parts[1].strip() if len(parts) > 1 else ''
        frac_v   = words_to_number(frac_raw.split()) if frac_raw else 0.0
        if frac_v >= 1:
            digits = math.floor(math.log10(frac_v)) + 1
            frac_v = frac_v / (10 ** digits)
        v = whole + frac_v
        return {'value': round(v, 6), 'unit': 'fraction', 'formatted': str(round(v, 4))}
    if 'nuqta' in raw_l:
        num_toks = raw_l.replace('nuqta', '').split()
        frac_v   = words_to_number(num_toks)
        if frac_v >= 1:
            digits = math.floor(math.log10(frac_v)) + 1
            frac_v = frac_v / (10 ** digits)
        return {'value': round(frac_v, 6), 'unit': 'fraction', 'formatted': str(round(frac_v, 4))}
    v = words_to_number(raw_l.split())
    return {'value': v, 'unit': 'fraction', 'formatted': str(v)}


def _norm_ORD(raw: str) -> dict:
    raw_l = raw.lower().strip()
    for word, num in ORDINAL_MAP.items():
        if word in raw_l:
            return {'value': num, 'unit': 'ordinal', 'formatted': f"{num}-o'rinchi"}
    clean = re.sub(r'(inshi|ünshi|ınshı|inchi|chi)$', '', raw_l).strip()
    num   = int(words_to_number(clean.split())) if clean else 0
    return {'value': num, 'unit': 'ordinal', 'formatted': f"{num}-o'rinchi"}


def _norm_APX(raw: str) -> dict:
    raw_l = raw.lower().strip()
    for phrase, meaning in sorted(APPROX_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in raw_l:
            return {'value': meaning, 'unit': 'approx', 'formatted': meaning}
    return {'value': '~?', 'unit': 'approx', 'formatted': raw}


_NORMALIZERS = {
    'MON': _norm_MON, 'PCT': _norm_PCT, 'DAT': _norm_DAT,
    'TIM': _norm_TIM, 'CNT': _norm_CNT, 'FRC': _norm_FRC,
    'ORD': _norm_ORD, 'APX': _norm_APX,
}


# =============================================================================
# 8. OVERLAP RESOLUTION
# =============================================================================

def _resolve_overlaps(matches: list) -> list:
    """
    Keep the best non-overlapping set.
    Tie-break: higher confidence > longer span > earlier in priority list.
    """
    matches.sort(key=lambda x: (
        x['start'],
        -(x['end'] - x['start']),
        -x.get('confidence', 0),
        _PRIORITY.index(x['type']) if x['type'] in _PRIORITY else 99,
    ))
    resolved, last_end = [], -1
    for m in matches:
        if m['start'] >= last_end:
            resolved.append(m)
            last_end = m['end']
        elif resolved:
            # Replace if significantly more confident
            if m.get('confidence', 0) > resolved[-1].get('confidence', 0) + 0.15:
                resolved[-1] = m
                last_end     = m['end']
    return resolved


# =============================================================================
# 9. SENTENCE INDEX HELPER
# =============================================================================

def _build_sent_offsets(text: str) -> list:
    sentences, offsets, pos = re.split(r'(?<=[.!?])\s+', text), [], 0
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
    """Remove duplicates, filter trivially weak APX, re-index."""
    seen, final = set(), []
    for r in sorted(results, key=lambda x: (x.sent_idx, x.start)):
        key = (r.type, r.raw.lower(), r.sent_idx)
        if key in seen:
            r.debug_trace.append('removed_duplicate')
            continue
        seen.add(key)
        # Drop single-word APX with low confidence (likely false positive)
        if r.type == 'APX' and r.confidence < 0.35 and len(r.raw.split()) == 1:
            r.debug_trace.append(f'filtered_weak_apx:{r.confidence:.2f}')
            continue
        final.append(r)
    for i, r in enumerate(final, 1):
        r.id = i
    return final


# =============================================================================
# 11. MAIN EXTRACT FUNCTION
# =============================================================================

def extract(text: str, debug: bool = False) -> list:
    """
    Full pipeline. Returns list of NERResult objects.

    Args:
        text:  Raw input (any spelling, any script variant)
        debug: Populate debug_trace on every result + emit log messages
    """
    normalized   = normalize_text(text)
    sent_offsets = _build_sent_offsets(normalized)

    if debug:
        logger.debug("Normalized: %r", normalized)

    # ── Pattern matching ──────────────────────────────────
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
                'type': etype, 'raw': span, 'start': start, 'end': end,
                'confidence': conf, 'signals': sigs,
                'ctx_before': ctx_b, 'ctx_after': ctx_a,
            })

    resolved = _resolve_overlaps(raw_matches)

    # ── Normalize + validate ──────────────────────────────
    results = []
    for idx, match in enumerate(resolved):
        etype, raw, trace = match['type'], match['raw'], []
        if debug:
            trace += [f"pattern:{etype}", f"signals:{','.join(match['signals'])}"]

        normalizer = _NORMALIZERS.get(etype)
        try:
            norm = normalizer(raw) if normalizer else {}
        except Exception as exc:
            norm = {'value': raw, 'unit': '', 'formatted': raw}
            trace.append(f"normalizer_error:{exc}")
            logger.warning("Normalizer error %s %r: %s", etype, raw, exc)

        value, unit, formatted = norm.get('value',''), norm.get('unit',''), norm.get('formatted', raw)

        valid, reason = _validate_span(etype, value, raw)
        if not valid:
            trace.append(f"validation_failed:{reason}")
            match['signals'].append('value_out_of_range')
            match['confidence'] = max(0.0, match['confidence'] + _SIGNAL_WEIGHTS['value_out_of_range'])
            if debug:
                logger.debug("Validation failed %s %r: %s", etype, raw, reason)
        elif debug:
            trace.append(f"valid:{reason}")

        results.append(NERResult(
            id=idx + 1, type=etype, raw=raw, formatted=formatted,
            value=value, unit=unit,
            confidence=match['confidence'], signals=match['signals'],
            sent_idx=_get_sent_idx(match['start'], sent_offsets),
            start=match['start'], end=match['end'],
            debug_trace=trace,
        ))

    return _post_process(results)


# =============================================================================
# 12. PUBLIC API
# =============================================================================

def extract_from_sentences(text: str, debug: bool = False) -> tuple:
    """
    Main public function used by Django views.
    Returns (results_as_dicts, sentences, token_count).
    """
    sentences   = re.split(r'(?<=[.!?])\s+', text.strip())
    ner_results = extract(text, debug=debug)
    token_count = len(text.split())
    return [r.to_dict() for r in ner_results], sentences, token_count


def extract_rich(text: str, debug: bool = False) -> tuple:
    """
    Like extract_from_sentences but returns NERResult objects.
    Use when you need confidence scores and signals in your own code.
    """
    sentences   = re.split(r'(?<=[.!?])\s+', text.strip())
    ner_results = extract(text, debug=debug)
    return ner_results, sentences, len(text.split())


def stats(results: list) -> dict:
    """Summary statistics over a list of result dicts."""
    if not results:
        return {'total': 0, 'by_type': {}, 'avg_confidence': 0.0}
    by_type     = {}
    confidences = []
    for r in results:
        t = r.get('type', 'UNK')
        by_type[t] = by_type.get(t, 0) + 1
        confidences.append(float(r.get('confidence', 0)))
    avg = sum(confidences) / len(confidences)
    return {
        'total':           len(results),
        'by_type':         by_type,
        'avg_confidence':  round(avg, 3),
        'high_confidence': sum(1 for c in confidences if c >= 0.7),
        'low_confidence':  sum(1 for c in confidences if c < 0.5),
    }
