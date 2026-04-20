# ml/rule_ner.py
# Rule-based Named Entity Recognition for Karakalpak numerical expressions
# Covers 8 entity types: MON, PCT, DAT, TIM, CNT, FRC, ORD, APX

import re

# ─── Karakalpak number word dictionaries ──────────────────────────────────────

UNITS = {
    'bir': 1, 'eki': 2, 'üsh': 3, 'tört': 4, 'bes': 5,
    'altı': 6, 'jeti': 7, 'segiz': 8, 'toğız': 9, 'on': 10,
    'jigirma': 20, 'otiz': 30, 'qıriq': 40, 'eliw': 50,
    'alpıs': 60, 'jetpis': 70, 'seksen': 80, 'toqsan': 90,
    # digit strings handled separately
}

MULTIPLIERS = {
    'jüz': 100,
    'müng': 1_000,
    'million': 1_000_000,
    'milyard': 1_000_000_000,
}

SPECIAL_MULTIPLIERS = {
    'yarım': 0.5,
    'çerek': 0.25,
}

CURRENCY_WORDS = [
    "so'm", "som", "so`m", "tenge", "dollar", "rubl",
    "UZS", "KZT", "USD", "RUB",
]

PERCENT_WORDS = ['protsent', 'protcent', 'üleske', '%']

ORDINAL_MAP = {
    'birinshi': 1, 'birinchi': 1,
    'ekinshi': 2,  'ikkinchi': 2, 'ikinci': 2,
    'üshinshi': 3, 'üshinchi': 3, 'uchinchi': 3,
    'tötinshi': 4, 'törtinshi': 4,
    'beshinshi': 5, 'beshinchi': 5,
    'altınshı': 6, 'altinchi': 6,
    'jetinshi': 7, 'yettinchi': 7,
    'segizinshi': 8, 'sakkizinchi': 8,
    'toğızınshı': 9, 'toqqizinchi': 9,
    'oninshi': 10,  'o\'ninchi': 10,
}

MONTH_MAP = {
    # International names
    'yanvar': 1, 'fevral': 2, 'mart': 3, 'aprel': 4,
    'may': 5, 'iyun': 6, 'iyul': 7, 'avgust': 8,
    'sentabr': 9, 'sentabır': 9, 'oktabr': 10,
    'noyabr': 11, 'dekabr': 12,
    # Karakalpak national month names
    'qantar': 1, 'aqpan': 2, 'naurız': 3, 'kokirek': 4,
    'mamır': 5, 'mavsım': 6, 'shilde': 7, 'tamız': 8,
    'qırküyrek': 9, 'qazan': 10, 'noabır': 11, 'jeltoqsan': 12,
}

APPROX_MAP = {
    'bir neshe': '~3–5',
    'birneshe': '~3–5',
    'birqansha': '~5–10',
    'bir qansha': '~5–10',
    'az': 'az miqdor (a little)',
    'köp': "ko'p (many)",
    'öte köp': "juda ko'p (very many)",
    'tolıp': "ko'p (many)",
}

TIME_CONTEXT = {
    'tüste': '12:00–14:00',
    'tuste': '12:00–14:00',
    'keshte': '18:00–21:00',
    'erteng': '06:00–10:00',
    'ertalab': '06:00–10:00',
    'tünde': '22:00–06:00',
    'tunde': '22:00–06:00',
}

# ─── Build number-word pattern string (for use in regex) ─────────────────────

_all_num_words = (
    list(UNITS.keys()) +
    list(MULTIPLIERS.keys()) +
    list(SPECIAL_MULTIPLIERS.keys())
)
_NUM_WORD_PAT = '|'.join(sorted(_all_num_words, key=len, reverse=True))
_MONTH_PAT    = '|'.join(sorted(MONTH_MAP.keys(), key=len, reverse=True))
_ORD_PAT      = '|'.join(sorted(ORDINAL_MAP.keys(), key=len, reverse=True))
_CURR_PAT     = '|'.join(re.escape(c) for c in sorted(CURRENCY_WORDS, key=len, reverse=True))
_PCT_PAT      = '|'.join(re.escape(p) for p in sorted(PERCENT_WORDS, key=len, reverse=True))
_APPROX_PAT   = '|'.join(re.escape(a) for a in sorted(APPROX_MAP.keys(), key=len, reverse=True))
_TIME_CTX_PAT = '|'.join(re.escape(t) for t in sorted(TIME_CONTEXT.keys(), key=len, reverse=True))

# ─── Compiled regex patterns ──────────────────────────────────────────────────

def _build_patterns():
    """Build and return all compiled regex patterns."""

    # MON — monetary expressions
    mon_pat = (
        r'(?:'
        r'(?:\d[\d\s]*(?:müng|million|milyard)?)'
        r'|'
        r'(?:(?:yarım|çerek)\s+(?:müng|million|milyard))'
        r'|'
        r'(?:(?:' + _NUM_WORD_PAT + r')\s*(?:(?:' + _NUM_WORD_PAT + r')\s*)*)'
        r')'
        r'\s*(?:' + _CURR_PAT + r')'
    )

    # PCT — percentage
    pct_pat = (
        r'(?:'
        r'\d+(?:[.,]\d+)?\s*%'
        r'|'
        r'(?:\d+|(?:' + _NUM_WORD_PAT + r')(?:\s+(?:' + _NUM_WORD_PAT + r'))*)'
        r'\s+(?:' + _PCT_PAT + r')'
        r')'
    )

    # DAT — dates
    dat_pat = (
        r'(?:'
        r'\d{4}[-\s]jılı(?:\s+\d{1,2}[-\s](?:' + _MONTH_PAT + r'))?'
        r'|'
        r'\d{1,2}[-\s.](?:' + _MONTH_PAT + r')(?:[-\s.]\d{2,4})?'
        r'|'
        r'(?:' + _MONTH_PAT + r')\s+\d{1,2}(?:[-\s,]\s*\d{2,4})?'
        r'|'
        r'\d{1,2}[./]\d{1,2}[./]\d{2,4}'
        r')'
    )

    # TIM — time expressions
    tim_pat = (
        r'(?:'
        r'saat\s+(?:\d{1,2}(?::\d{2})?|(?:' + _NUM_WORD_PAT + r')(?:\s+(?:' + _NUM_WORD_PAT + r'))*)'
        r'|'
        r'(?:' + _TIME_CTX_PAT + r')'
        r')'
    )

    # ORD — ordinal numbers
    ord_pat = r'\b(?:' + _ORD_PAT + r')\b'

    # CNT — count/quantity expressions
    cnt_pat = (
        r'(?:'
        r'(?:\d[\d\s,.]*)'
        r'|'
        r'(?:(?:' + _NUM_WORD_PAT + r')(?:\s+(?:' + _NUM_WORD_PAT + r'))*)'
        r')'
        r'\s+'
        r'(?:gektar|kisi|adam|dana|baş|metr|km|litr|tonna|sm|m2|km2|soat|minut|sekund|yıl|ay|kün)'
    )

    # FRC — fractional numbers
    frc_pat = (
        r'(?:'
        r'\d+[.,]\d+'
        r'|'
        r'(?:(?:' + _NUM_WORD_PAT + r')\s+)*butın(?:\s+(?:' + _NUM_WORD_PAT + r'))+'
        r'|'
        r'nuqta\s+(?:' + _NUM_WORD_PAT + r')(?:\s+(?:' + _NUM_WORD_PAT + r'))*'
        r')'
    )

    # APX — approximate / vague quantities
    apx_pat = r'\b(?:' + _APPROX_PAT + r')\b'

    flags = re.IGNORECASE | re.UNICODE
    return {
        'MON': re.compile(mon_pat, flags),
        'PCT': re.compile(pct_pat, flags),
        'DAT': re.compile(dat_pat, flags),
        'TIM': re.compile(tim_pat, flags),
        'ORD': re.compile(ord_pat, flags),
        'CNT': re.compile(cnt_pat, flags),
        'FRC': re.compile(frc_pat, flags),
        'APX': re.compile(apx_pat, flags),
    }


PATTERNS = _build_patterns()

# ─── words_to_number ──────────────────────────────────────────────────────────

def words_to_number(tokens: list) -> float:
    """Convert a list of Karakalpak number word tokens to a float."""
    result  = 0.0
    current = 0.0

    for tok in tokens:
        tl = tok.lower().strip(".,;:!?\"'")

        # Plain digit
        if re.fullmatch(r'\d+', tl):
            current += int(tl)
            continue

        # Decimal digit
        if re.fullmatch(r'\d+[.,]\d+', tl):
            current += float(tl.replace(',', '.'))
            continue

        if tl in UNITS:
            current += UNITS[tl]
        elif tl in MULTIPLIERS:
            mult = MULTIPLIERS[tl]
            if mult >= 1000:
                result  += (current if current else 1) * mult
                current  = 0.0
            else:
                current  = (current if current else 1) * mult
        elif tl in SPECIAL_MULTIPLIERS:
            current = (current if current else 1) * SPECIAL_MULTIPLIERS[tl]

    return result + current

# ─── Normalizer functions (one per entity type) ───────────────────────────────

def normalize_MON(raw: str) -> dict:
    tokens   = raw.split()
    currency = 'UZS'
    num_toks = []
    for tok in tokens:
        tl = tok.lower().strip(".,")
        if tl in [c.lower() for c in CURRENCY_WORDS]:
            # map to standard code
            mapping = {
                "so'm": 'UZS', "som": 'UZS', "so`m": 'UZS',
                'tenge': 'KZT', 'dollar': 'USD', 'rubl': 'RUB',
                'uzs': 'UZS', 'kzt': 'KZT', 'usd': 'USD', 'rub': 'RUB',
            }
            currency = mapping.get(tl, tl.upper())
        else:
            num_toks.append(tok)
    value = words_to_number(num_toks)
    return {
        'value':     value,
        'unit':      currency,
        'formatted': f"{value:,.0f} {currency}",
    }


def normalize_PCT(raw: str) -> dict:
    # Handle "85%" directly
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*%', raw)
    if m:
        value = float(m.group(1).replace(',', '.'))
        return {'value': value, 'unit': '%', 'formatted': f"{value:.1f}%"}

    tokens   = raw.split()
    num_toks = [t for t in tokens
                if t.lower() not in ('protsent', 'protcent', 'üleske', '%')]

    # "beshten bir üleske" → 1/5 = 20%
    for i, tok in enumerate(tokens):
        if tok.lower().endswith(('ten', 'tan', 'den', 'dan')) and i + 1 < len(tokens):
            base_word = tok[:-3]
            base      = words_to_number([base_word])
            if base:
                value = round(100 / base, 2)
                return {'value': value, 'unit': '%', 'formatted': f"{value:.1f}%"}

    value = words_to_number(num_toks)
    return {'value': value, 'unit': '%', 'formatted': f"{value:.1f}%"}


def normalize_DAT(raw: str) -> dict:
    raw_l = raw.lower().strip()
    day, month, year = None, None, None

    # dd.mm.yyyy or dd/mm/yyyy
    m = re.match(r'(\d{1,2})[./](\d{1,2})[./](\d{2,4})', raw_l)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        return {'value': f"{year}-{month:02d}-{day:02d}",
                'unit': 'date', 'formatted': f"{day:02d}.{month:02d}.{year}"}

    # year with -jılı
    m = re.search(r'(\d{4})[-\s]jılı', raw_l)
    if m:
        year = int(m.group(1))

    # day number
    m = re.search(r'\b(\d{1,2})\b', raw_l)
    if m:
        day = int(m.group(1))

    # month name
    for name, num in sorted(MONTH_MAP.items(), key=lambda x: -len(x[0])):
        if name in raw_l:
            month = num
            break

    y = year  or 2024
    mo = month or 1
    d  = day   or 1
    return {
        'value':     f"{y}-{mo:02d}-{d:02d}",
        'unit':      'date',
        'formatted': f"{d:02d}.{mo:02d}.{y}",
    }


def normalize_TIM(raw: str) -> dict:
    raw_l = raw.lower().strip()

    # Context words
    for word, rng in TIME_CONTEXT.items():
        if word in raw_l:
            return {'value': rng, 'unit': 'time_range', 'formatted': rng}

    # "saat HH:MM"
    m = re.search(r'(\d{1,2}):(\d{2})', raw_l)
    if m:
        h, mn = int(m.group(1)), int(m.group(2))
        return {'value': f"{h:02d}:{mn:02d}", 'unit': 'time',
                'formatted': f"{h:02d}:{mn:02d}"}

    # "saat on eki"
    tokens   = raw_l.split()
    num_toks = [t for t in tokens if t != 'saat']
    hour     = int(words_to_number(num_toks)) if num_toks else 0
    return {'value': f"{hour:02d}:00", 'unit': 'time',
            'formatted': f"{hour:02d}:00"}


def normalize_CNT(raw: str) -> dict:
    UNITS_OF_MEASURE = {
        'gektar', 'kisi', 'adam', 'dana', 'baş', 'metr', 'km',
        'litr', 'tonna', 'sm', 'm2', 'km2', 'soat', 'minut',
        'sekund', 'yıl', 'ay', 'kün',
    }
    tokens   = raw.split()
    unit     = ''
    num_toks = []
    for tok in tokens:
        if tok.lower() in UNITS_OF_MEASURE:
            unit = tok
        else:
            num_toks.append(tok)
    value = words_to_number(num_toks)
    fmt   = f"{value:,.0f} {unit}".strip()
    return {'value': value, 'unit': unit, 'formatted': fmt}


def normalize_FRC(raw: str) -> dict:
    raw_l = raw.lower().strip()

    # plain decimal  1.5 / 3,75
    m = re.match(r'^(\d+)[.,](\d+)$', raw_l)
    if m:
        value = float(raw_l.replace(',', '.'))
        return {'value': value, 'unit': 'fraction', 'formatted': str(value)}

    # "bir butın bes"
    if 'butın' in raw_l:
        parts     = raw_l.split('butın')
        whole     = words_to_number(parts[0].split())
        frac_part = parts[1].strip() if len(parts) > 1 else ''
        frac_val  = words_to_number(frac_part.split()) if frac_part else 0
        # normalize frac_val: 5→0.5, 25→0.25, 75→0.75
        if frac_val >= 1:
            import math
            digits = math.floor(math.log10(frac_val)) + 1
            frac_val = frac_val / (10 ** digits)
        value = whole + frac_val
        return {'value': value, 'unit': 'fraction', 'formatted': str(round(value, 4))}

    # "nuqta toqsan bes" → 0.95
    if 'nuqta' in raw_l:
        tokens    = raw_l.replace('nuqta', '').split()
        frac_val  = words_to_number(tokens)
        if frac_val >= 1:
            import math
            digits   = math.floor(math.log10(frac_val)) + 1
            frac_val = frac_val / (10 ** digits)
        return {'value': frac_val, 'unit': 'fraction', 'formatted': str(round(frac_val, 4))}

    value = words_to_number(raw_l.split())
    return {'value': value, 'unit': 'fraction', 'formatted': str(value)}


def normalize_ORD(raw: str) -> dict:
    raw_l = raw.lower().strip()
    for word, num in ORDINAL_MAP.items():
        if word in raw_l:
            return {'value': num, 'unit': 'ordinal',
                    'formatted': f"{num}-o'rinchi"}
    # fallback: strip suffix and parse
    clean = re.sub(r'(inshi|ünshi|ınshı|inchi|chi)$', '', raw_l).strip()
    num   = int(words_to_number(clean.split())) if clean else 0
    return {'value': num, 'unit': 'ordinal', 'formatted': f"{num}-o'rinchi"}


def normalize_APX(raw: str) -> dict:
    raw_l = raw.lower().strip()
    for phrase, meaning in sorted(APPROX_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in raw_l:
            return {'value': meaning, 'unit': 'approx', 'formatted': meaning}
    return {'value': '~?', 'unit': 'approx', 'formatted': raw}


NORMALIZERS = {
    'MON': normalize_MON,
    'PCT': normalize_PCT,
    'DAT': normalize_DAT,
    'TIM': normalize_TIM,
    'CNT': normalize_CNT,
    'FRC': normalize_FRC,
    'ORD': normalize_ORD,
    'APX': normalize_APX,
}
# ─── Compound word splitter ───────────────────────────────────────────────────

_ALL_NUM_WORDS = sorted(
    list(UNITS.keys()) +
    list(MULTIPLIERS.keys()) +
    list(SPECIAL_MULTIPLIERS.keys()) +
    list(ORDINAL_MAP.keys()) +
    list(MONTH_MAP.keys()) +
    CURRENCY_WORDS +
    PERCENT_WORDS +
    ['saat', 'butın', 'nuqta', 'dana', 'gektar', 'kisi', 'adam',
     'baş', 'metr', 'tonna', 'litr', 'protsent', 'üleske',
     'müng', 'million', 'milyard', 'jılı', 'neshe', 'qansha'],
    key=len,
    reverse=True
)

_KNOWN_WORD_RE = re.compile(
    r'^(' + '|'.join(re.escape(w) for w in _ALL_NUM_WORDS) + r')',
    re.IGNORECASE | re.UNICODE
)


def _split_compound(token: str) -> str:
    tl = token.lower()
    if tl in [w.lower() for w in _ALL_NUM_WORDS]:
        return token
    remaining = tl
    parts = []
    while remaining:
        m = _KNOWN_WORD_RE.match(remaining)
        if m:
            parts.append(m.group(1))
            remaining = remaining[m.end():]
        else:
            return token
    if not remaining:
        return ' '.join(parts)
    return token


def normalize_text(text: str) -> str:
    text = text.replace('so`m', "so'm")
    text = re.sub(r'\bpratsent\b', 'protsent', text, flags=re.IGNORECASE)
    text = re.sub(r'\bprotcent\b', 'protsent', text, flags=re.IGNORECASE)
    tokens = re.split(r'(\s+)', text)
    result_tokens = []
    for tok in tokens:
        if re.match(r'\s+', tok):
            result_tokens.append(tok)
        else:
            result_tokens.append(_split_compound(tok))
    return ''.join(result_tokens)


# ─── Main extraction function ─────────────────────────────────────────────────

def extract(text: str) -> list:
    normalized = normalize_text(text)
    raw_matches = []
    priority = ['MON', 'DAT', 'TIM', 'PCT', 'CNT', 'FRC', 'ORD', 'APX']

    for etype in priority:
        pattern = PATTERNS[etype]
        for m in pattern.finditer(normalized):
            raw_matches.append({
                'type':  etype,
                'raw':   m.group().strip(),
                'start': m.start(),
                'end':   m.end(),
            })

    raw_matches.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))
    resolved = []
    last_end = -1
    for match in raw_matches:
        if match['start'] >= last_end:
            resolved.append(match)
            last_end = match['end']

    sentences = re.split(r'(?<=[.!?])\s+', normalized)
    sent_offsets = []
    pos = 0
    for i, s in enumerate(sentences):
        sent_offsets.append((pos, pos + len(s), i))
        pos += len(s) + 1

    def get_sent_idx(start):
        for s_start, s_end, idx in sent_offsets:
            if s_start <= start < s_end:
                return idx
        return 0

    results = []
    for idx, match in enumerate(resolved):
        etype = match['type']
        raw   = match['raw']
        if not raw.strip():
            continue
        normalizer = NORMALIZERS.get(etype)
        if normalizer:
            try:
                norm = normalizer(raw)
            except Exception:
                norm = {'value': raw, 'unit': '', 'formatted': raw}
        else:
            norm = {'value': raw, 'unit': '', 'formatted': raw}

        results.append({
            'id':        idx + 1,
            'type':      etype,
            'raw':       raw,
            'start':     match['start'],
            'end':       match['end'],
            'value':     norm.get('value', ''),
            'unit':      norm.get('unit', ''),
            'formatted': norm.get('formatted', raw),
            'sent_idx':  get_sent_idx(match['start']),
        })

    return results


def extract_from_sentences(text: str) -> tuple:
    """Return (results, sentences, token_count)."""
    sentences   = re.split(r'(?<=[.!?])\s+', text.strip())
    all_results = extract(text)
    token_count = len(text.split())
    return all_results, sentences, token_count