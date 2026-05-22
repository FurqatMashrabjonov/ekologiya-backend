"""
Shared utilities for TTS text preprocessing.
- Uzbek number → ordinal text (any number up to 999 999)
- Uzbek Cyrillic → Latin transliteration
"""
import re


# ── Uzbek number → text ───────────────────────────────────────────────

_UZ_ONES = {
    0: '', 1: 'bir', 2: 'ikki', 3: 'uch', 4: "to'rt", 5: 'besh',
    6: 'olti', 7: 'yetti', 8: 'sakkiz', 9: "to'qqiz"
}

_UZ_TENS = {
    1: "o'n", 2: 'yigirma', 3: "o'ttiz", 4: 'qirq', 5: 'ellik',
    6: 'oltmish', 7: 'yetmish', 8: 'sakson', 9: "to'qson"
}

_UZ_ORDINAL_MAP = {
    'bir': 'birinchi', 'ikki': 'ikkinchi', 'uch': 'uchinchi',
    "to'rt": "to'rtinchi", 'besh': 'beshinchi', 'olti': 'oltinchi',
    'yetti': 'yettinchi', 'sakkiz': 'sakkizinchi', "to'qqiz": "to'qqizinchi",
    "o'n": "o'ninchi", 'yigirma': 'yigirmanchi', "o'ttiz": "o'ttizinchi",
    'qirq': 'qirqinchi', 'ellik': 'ellikinchi', 'oltmish': 'oltmishinchi',
    'yetmish': 'yetmishinchi', 'sakson': 'saksoninchi', "to'qson": "to'qsoninchi",
    'yuz': 'yuzinchi', 'ming': 'minginchi'
}


def uz_number_to_text(n: int) -> str:
    """Convert integer to Uzbek cardinal text.  0–999 999."""
    if n == 0:
        return 'nol'

    parts = []

    if n >= 1000:
        thousands = n // 1000
        if thousands == 1:
            parts.append('ming')
        else:
            parts.append(uz_number_to_text(thousands) + ' ming')
        n %= 1000

    if n >= 100:
        hundreds = n // 100
        if hundreds == 1:
            parts.append('yuz')
        else:
            parts.append(_UZ_ONES[hundreds] + ' yuz')
        n %= 100

    if n >= 10:
        parts.append(_UZ_TENS[n // 10])
        n %= 10

    if n > 0:
        parts.append(_UZ_ONES[n])

    return ' '.join(parts)


def uz_number_to_ordinal(n: int) -> str:
    """Convert integer to Uzbek ordinal text.  43 → qirq uchinchi."""
    cardinal = uz_number_to_text(n)
    words = cardinal.split()
    last = words[-1]
    words[-1] = _UZ_ORDINAL_MAP.get(last, last + 'inchi')
    return ' '.join(words)


# ── Russian number → text ─────────────────────────────────────────────

_RU_ONES = {
    0: '', 1: 'один', 2: 'два', 3: 'три', 4: 'четыре', 5: 'пять',
    6: 'шесть', 7: 'семь', 8: 'восемь', 9: 'девять'
}

_RU_TEENS = {
    10: 'десять', 11: 'одиннадцать', 12: 'двенадцать', 13: 'тринадцать',
    14: 'четырнадцать', 15: 'пятнадцать', 16: 'шестнадцать', 17: 'семнадцать',
    18: 'восемнадцать', 19: 'девятнадцать'
}

_RU_TENS = {
    2: 'двадцать', 3: 'тридцать', 4: 'сорок', 5: 'пятьдесят',
    6: 'шестьдесят', 7: 'семьдесят', 8: 'восемьдесят', 9: 'девяносто'
}

_RU_HUNDREDS = {
    1: 'сто', 2: 'двести', 3: 'триста', 4: 'четыреста', 5: 'пятьсот',
    6: 'шестьсот', 7: 'семьсот', 8: 'восемьсот', 9: 'девятьсот'
}

_RU_ORDINAL_MAP = {
    'один': 'первый', 'два': 'второй', 'три': 'третий', 'четыре': 'четвёртый',
    'пять': 'пятый', 'шесть': 'шестой', 'семь': 'седьмой', 'восемь': 'восьмой',
    'девять': 'девятый', 'десять': 'десятый',
    'одиннадцать': 'одиннадцатый', 'двенадцать': 'двенадцатый',
    'тринадцать': 'тринадцатый', 'четырнадцать': 'четырнадцатый',
    'пятнадцать': 'пятнадцатый', 'шестнадцать': 'шестнадцатый',
    'семнадцать': 'семнадцатый', 'восемнадцать': 'восемнадцатый',
    'девятнадцать': 'девятнадцатый',
    'двадцать': 'двадцатый', 'тридцать': 'тридцатый', 'сорок': 'сороковой',
    'пятьдесят': 'пятидесятый', 'шестьдесят': 'шестидесятый',
    'семьдесят': 'семидесятый', 'восемьдесят': 'восьмидесятый',
    'девяносто': 'девяностый',
    'сто': 'сотый', 'двести': 'двухсотый', 'триста': 'трёхсотый',
    'четыреста': 'четырёхсотый', 'пятьсот': 'пятисотый',
    'шестьсот': 'шестисотый', 'семьсот': 'семисотый',
    'восемьсот': 'восьмисотый', 'девятьсот': 'девятисотый',
    'тысяча': 'тысячный',
}


def ru_number_to_text(n: int) -> str:
    """Convert integer to Russian cardinal text.  0–999 999."""
    if n == 0:
        return 'ноль'

    parts = []

    if n >= 1000:
        t = n // 1000
        if t == 1:
            parts.append('тысяча')
        elif t == 2:
            parts.append('две тысячи')
        elif 3 <= t <= 4:
            parts.append(_RU_ONES[t] + ' тысячи')
        elif 5 <= t <= 20:
            # 5-20 → "пять тысяч", handle teens too
            sub = _RU_TEENS.get(t, _RU_ONES.get(t, ''))
            parts.append(sub + ' тысяч')
        else:
            parts.append(ru_number_to_text(t) + ' тысяч')
        n %= 1000

    if n >= 100:
        parts.append(_RU_HUNDREDS[n // 100])
        n %= 100

    if 10 <= n <= 19:
        parts.append(_RU_TEENS[n])
        n = 0
    elif n >= 20:
        parts.append(_RU_TENS[n // 10])
        n %= 10

    if n > 0:
        parts.append(_RU_ONES[n])

    return ' '.join(parts)


def ru_number_to_ordinal(n: int) -> str:
    """Convert integer to Russian ordinal (masculine).  2020 → две тысячи двадцатый."""
    cardinal = ru_number_to_text(n)
    words = cardinal.split()
    last = words[-1]
    words[-1] = _RU_ORDINAL_MAP.get(last, last + '-й')
    return ' '.join(words)


# ── Uzbek Cyrillic → Latin transliteration ────────────────────────────

# Multi-char Cyrillic letters → Latin (applied first)
_CYR_MULTI = [
    ('Ш', 'Sh'), ('ш', 'sh'),
    ('Ч', 'Ch'), ('ч', 'ch'),
    ('Ю', 'Yu'), ('ю', 'yu'),
    ('Я', 'Ya'), ('я', 'ya'),
    ('Ё', 'Yo'), ('ё', 'yo'),
    ('Ц', 'Ts'), ('ц', 'ts'),
    ('Ғ', "G'"), ('ғ', "g'"),
    ('Ў', "O'"), ('ў', "o'"),
]

# Single-char Cyrillic → Latin
_CYR_SINGLE = {
    'А': 'A', 'а': 'a', 'Б': 'B', 'б': 'b', 'В': 'V', 'в': 'v',
    'Г': 'G', 'г': 'g', 'Д': 'D', 'д': 'd', 'Е': 'E', 'е': 'e',
    'Ж': 'J', 'ж': 'j', 'З': 'Z', 'з': 'z', 'И': 'I', 'и': 'i',
    'Й': 'Y', 'й': 'y', 'К': 'K', 'к': 'k', 'Л': 'L', 'л': 'l',
    'М': 'M', 'м': 'm', 'Н': 'N', 'н': 'n', 'О': 'O', 'о': 'o',
    'П': 'P', 'п': 'p', 'Р': 'R', 'р': 'r', 'С': 'S', 'с': 's',
    'Т': 'T', 'т': 't', 'У': 'U', 'у': 'u', 'Ф': 'F', 'ф': 'f',
    'Х': 'X', 'х': 'x', 'Э': 'E', 'э': 'e',
    'Қ': 'Q', 'қ': 'q', 'Ҳ': 'H', 'ҳ': 'h',
    'Ъ': "'", 'ъ': "'",
    'Ь': '', 'ь': '',
}

_HAS_CYRILLIC = re.compile(r'[а-яА-ЯўқғҳЎҚҒҲёЁ]')


def cyrillic_to_latin(text: str) -> str:
    """Transliterate Uzbek Cyrillic → Latin for TTS.
    Leaves non-Cyrillic parts (digits, punctuation, Latin) unchanged.
    """
    if not _HAS_CYRILLIC.search(text):
        return text

    for cyr, lat in _CYR_MULTI:
        text = text.replace(cyr, lat)

    return ''.join(_CYR_SINGLE.get(ch, ch) for ch in text)
