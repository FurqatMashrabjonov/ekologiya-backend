"""
3-tier hybrid language detector for Russian / Uzbek.

Tier 1: Script analysis (instant) — Uzbek-specific Cyrillic chars
Tier 2: FastText ML model (<1ms) — trained on millions of texts, 176 languages
Tier 3: Keyword fallback (~0ms) — extended word lists for ambiguous cases
"""

import re
import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

# FastText model path
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "lid.176.ftz"
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

# Lazy-loaded FastText model
_ft_model = None


def _ensure_model():
    """Download FastText lid.176.ftz if not present (~900KB compressed)."""
    global _ft_model
    if _ft_model is not None:
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"📥 [LANG] Downloading FastText lid.176.ftz model...")
        try:
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            print(f"✅ [LANG] Model downloaded to {MODEL_PATH}")
        except Exception as e:
            print(f"❌ [LANG] Failed to download model: {e}")
            return

    try:
        import fasttext
        # Suppress FastText warning about loading with 'loadModel'
        _ft_model = fasttext.load_model(str(MODEL_PATH))
        print(f"✅ [LANG] FastText model loaded ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")
    except Exception as e:
        print(f"❌ [LANG] Failed to load FastText model: {e}")


# --- Uzbek-specific Cyrillic characters (don't exist in Russian) ---
UZ_CYRL_CHARS = set("ўқҳғЎҚҲҒ")

# --- Extended keyword dictionaries for Tier 3 fallback ---

# Common Russian words in Latin transliteration (not typically Uzbek)
RU_LATIN_WORDS = {
    "kak", "chto", "gde", "kogda", "pochemu", "zachem", "skolko",
    "mozhno", "nelzya", "da", "net", "eto", "est", "bylo", "budet",
    "ya", "ty", "on", "ona", "my", "vy", "oni", "mne", "tebe",
    "privet", "zdravstvuyte", "spasibo", "pozhaluysta", "izvinite",
    "pomogite", "pomoshch", "vopros", "otvet", "zakon", "postanovlenie",
    "pravilo", "trebovanie", "razreshenie", "zapresheno", "razresheno",
    "kakie", "kakoy", "kakaya", "kakoe", "kotorye", "kotoryy",
    "dolzhen", "dolzhna", "nuzhno", "nado", "mozhete", "skazhite",
    "obyasnite", "rasskazhite", "podrobnee", "podskajite",
    "ekologiya", "ekspertiza", "priroda", "okruzhayushchaya", "sreda",
    "zagryaznenie", "vybros", "otkhodov", "shtraf", "narusheniye",
    "stroitelstvo", "predpriyatiye", "zavod", "fabrika",
    "kategoriya", "dokument", "litsenziya", "sertifikat",
    "chelovek", "lyudi", "rabota", "vremya", "den", "god",
    "pervaya", "vtoraya", "tretya", "chetvertaya", "pyataya",
    "skolko", "stoit", "tsena", "summa", "dengi",
    "pro", "pro", "dlya", "bez", "nad", "pod", "pered", "posle",
    "zdes", "tam", "tut", "seychas", "vchera", "segodnya", "zavtra",
    "bolshe", "menshe", "luchshe", "khuzhe", "ochen",
    "ili", "i", "no", "a", "chtoby", "yesli", "togda", "poetomu",
}

# Common Uzbek words in Latin (not typically Russian)
UZ_LATIN_WORDS = {
    "nima", "qanday", "qachon", "qayerda", "nega", "nimaga",
    "mumkin", "kerak", "bor", "yo'q", "yoq", "bilan", "uchun",
    "haqida", "bo'yicha", "boyicha", "asosida", "muvofiq",
    "salom", "assalomu", "alaykum", "rahmat", "kechirasiz",
    "yordam", "bering", "savol", "javob", "qonun", "qaror",
    "hujjat", "ekspertiza", "ekologik", "tabiat", "muhit",
    "ifloslanish", "chiqindi", "jarima", "buzilish",
    "qurilish", "korxona", "zavod", "fabrika",
    "toifa", "kategoriya", "hujjat", "litsenziya", "sertifikat",
    "odam", "kishilar", "ish", "vaqt", "kun", "yil",
    "birinchi", "ikkinchi", "uchinchi", "tortinchi", "beshinchi",
    "to'rtinchi", "qancha", "narx", "summa", "pul",
    "men", "sen", "siz", "u", "biz", "ular", "menga", "senga",
    "kiradi", "taalluqli", "quyidagi", "belgilangan", "tartibda",
    "ta'sir", "faoliyat", "obyekt", "loyiha", "manba",
    "masalan", "jumladan", "respublika", "davlat",
    "yoki", "va", "lekin", "ammo", "shuning", "agar", "shunda",
    "shu", "bu", "o'sha", "osha", "qilish", "berish", "olish",
    "tushuntiring", "ayting", "gapirib", "batafsil",
    "nimalar", "qaysi", "nechanchi", "togda",
}

# Russian Cyrillic words that are definitely NOT Uzbek
RU_CYRL_WORDS = {
    "что", "как", "где", "когда", "почему", "зачем", "сколько",
    "можно", "нельзя", "это", "есть", "было", "будет",
    "я", "ты", "он", "она", "мы", "вы", "они", "мне", "тебе",
    "привет", "здравствуйте", "спасибо", "пожалуйста", "извините",
    "помогите", "помощь", "вопрос", "ответ", "закон", "постановление",
    "правило", "требование", "разрешение", "запрещено", "разрешено",
    "какие", "какой", "какая", "какое", "которые", "который",
    "должен", "должна", "нужно", "надо", "можете", "скажите",
    "объясните", "расскажите", "подробнее", "подскажите",
    "экология", "экспертиза", "природа", "окружающая", "среда",
    "загрязнение", "выброс", "отходов", "штраф", "нарушение",
    "строительство", "предприятие", "завод", "фабрика",
    "категория", "документ", "лицензия", "сертификат",
}

# Uzbek Cyrillic words (with special chars ў/қ/ҳ/ғ or unique Uzbek words)
UZ_CYRL_WORDS = {
    "нима", "қандай", "қачон", "қаерда", "нега", "нимага",
    "мумкин", "керак", "бор", "йўқ", "билан", "учун",
    "ҳақида", "бўйича", "асосида", "мувофиқ",
    "салом", "ассалому", "алайкум", "раҳмат", "кечирасиз",
    "ёрдам", "беринг", "савол", "жавоб", "қонун", "қарор",
    "ҳужжат", "экспертиза", "экологик", "табиат", "муҳит",
    "ифлосланиш", "чиқинди", "жарима", "бузилиш",
    "қурилиш", "корхона", "тоифа", "категория",
    "одам", "кишилар", "иш", "вақт", "кун", "йил",
    "биринчи", "иккинчи", "учинчи", "тўртинчи", "бешинчи",
    "қанча", "нарх", "сумма", "пул",
    "мен", "сен", "сиз", "биз", "улар", "менга", "сенга",
    "киради", "тааллуқли", "қуйидаги", "белгиланган",
    "таъсир", "фаолият", "объект", "лойиҳа", "манба",
    "масалан", "жумладан", "республика", "давлат",
    "ёки", "ва", "лекин", "аммо", "шунинг", "агар", "шунда",
}


class LanguageResult:
    """Language detection result."""
    __slots__ = ("lang", "confidence", "tier")

    def __init__(self, lang: str, confidence: float, tier: int):
        self.lang = lang  # "RU", "UZ_LATN", "UZ_CYRL"
        self.confidence = confidence
        self.tier = tier  # 1, 2, or 3

    def __repr__(self):
        return f"LanguageResult(lang='{self.lang}', conf={self.confidence:.2f}, tier={self.tier})"


def _has_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters."""
    return bool(re.search(r'[а-яА-ЯёЁ]', text))


def _has_uz_cyrillic(text: str) -> bool:
    """Check if text contains Uzbek-specific Cyrillic chars."""
    return bool(UZ_CYRL_CHARS & set(text))


def _tier1_script(text: str) -> Optional[LanguageResult]:
    """Tier 1: Script analysis — instant, handles clear-cut cases."""
    if _has_uz_cyrillic(text):
        return LanguageResult("UZ_CYRL", 1.0, 1)
    return None


def _tier2_fasttext(text: str) -> Optional[LanguageResult]:
    """Tier 2: FastText ML model — <1ms, handles typos and ambiguity."""
    _ensure_model()
    if _ft_model is None:
        return None

    # Clean text for FastText (remove newlines, collapse whitespace)
    clean = re.sub(r'\s+', ' ', text.strip())
    if not clean:
        return None

    try:
        predictions = _ft_model.predict(clean, k=3)  # top 3 predictions
        labels = predictions[0]  # ['__label__uz', '__label__ru', ...]
        scores = predictions[1]  # [0.95, 0.03, ...]

        top_label = labels[0].replace("__label__", "")
        top_score = float(scores[0])

        if top_score < 0.4:
            return None  # too uncertain, fall through to Tier 3

        # Map FastText language code to our format
        has_cyrillic = _has_cyrillic(text)

        if top_label == "ru":
            return LanguageResult("RU", top_score, 2)
        elif top_label == "uz":
            if has_cyrillic:
                return LanguageResult("UZ_CYRL", top_score, 2)
            else:
                return LanguageResult("UZ_LATN", top_score, 2)
        elif top_label in ("kk", "ky", "tg", "tt", "az", "tr"):
            # Turkic/Central Asian languages often confused with Uzbek
            # If second prediction is uz or ru, use that
            if len(labels) > 1:
                second_label = labels[1].replace("__label__", "")
                second_score = float(scores[1])
                if second_label == "uz":
                    if has_cyrillic:
                        return LanguageResult("UZ_CYRL", second_score, 2)
                    else:
                        return LanguageResult("UZ_LATN", second_score, 2)
                elif second_label == "ru":
                    return LanguageResult("RU", second_score, 2)
            # Default to UZ_LATN for Turkic confusion
            if has_cyrillic:
                return LanguageResult("UZ_CYRL", top_score * 0.5, 2)
            else:
                return LanguageResult("UZ_LATN", top_score * 0.5, 2)
        else:
            # Unknown language — try to use script to decide
            if has_cyrillic:
                return LanguageResult("RU", 0.3, 2)
            return None  # fall through to Tier 3

    except Exception as e:
        print(f"⚠️ [LANG] FastText error: {e}")
        return None


def _tier3_keywords(text: str) -> LanguageResult:
    """Tier 3: Keyword matching fallback — handles remaining ambiguity."""
    text_lower = text.lower()
    words = set(re.findall(r'[a-zA-Zа-яА-ЯёЁўқҳғЎҚҲҒ\']+', text_lower))
    has_cyrillic = _has_cyrillic(text)

    if has_cyrillic:
        # Cyrillic text: count RU vs UZ cyrillic word matches
        ru_score = len(words & RU_CYRL_WORDS)
        uz_score = len(words & UZ_CYRL_WORDS)

        if uz_score > ru_score:
            return LanguageResult("UZ_CYRL", 0.6, 3)
        elif ru_score > uz_score:
            return LanguageResult("RU", 0.6, 3)
        else:
            # Tie → default to RU for Cyrillic (more common case)
            return LanguageResult("RU", 0.4, 3)
    else:
        # Latin text: count RU-translit vs UZ-latin word matches
        ru_score = len(words & RU_LATIN_WORDS)
        uz_score = len(words & UZ_LATIN_WORDS)

        if uz_score > ru_score:
            return LanguageResult("UZ_LATN", 0.6, 3)
        elif ru_score > uz_score:
            return LanguageResult("RU", 0.6, 3)
        else:
            # Tie → default to UZ_LATN (primary audience)
            return LanguageResult("UZ_LATN", 0.4, 3)


def detect_language(text: str) -> LanguageResult:
    """
    Detect language of text using 3-tier hybrid approach.

    Returns LanguageResult with:
    - lang: "RU", "UZ_LATN", or "UZ_CYRL"
    - confidence: 0.0 - 1.0
    - tier: which tier decided (1, 2, or 3)
    """
    if not text or not text.strip():
        return LanguageResult("UZ_LATN", 0.0, 3)

    # Tier 1: Script analysis
    result = _tier1_script(text)
    if result:
        print(f"🔤 [LANG] Tier 1 (script): {result}")
        return result

    # Tier 2: FastText ML
    result = _tier2_fasttext(text)
    if result and result.confidence >= 0.4:
        print(f"🔤 [LANG] Tier 2 (FastText): {result}")
        return result

    # Tier 3: Keyword fallback
    result = _tier3_keywords(text)
    print(f"🔤 [LANG] Tier 3 (keywords): {result}")
    return result



# Turkic language labels recognized by FastText that indicate Uzbek/related content
_TURKIC_LABELS = {"uz", "tr", "az", "kk", "ky", "tt", "tg", "ug", "ba"}


def _uzbek_score(text: str, ft_model) -> float:
    """
    Compute how 'Uzbek-like' a text is, using FastText.
    Returns 0.0–1.0 score. Higher = more likely Uzbek.
    """
    if not text.strip():
        return 0.0
    try:
        pred = ft_model.predict(re.sub(r'\s+', ' ', text.strip()), k=3)
        labels = [l.replace("__label__", "") for l in pred[0]]
        confs = [float(c) for c in pred[1]]
        # Sum confidence for Turkic labels
        score = sum(c for l, c in zip(labels, confs) if l in _TURKIC_LABELS)
        return score
    except Exception:
        return 0.0


def _russian_score(text: str, ft_model) -> float:
    """
    Compute how 'Russian-like' a text is, using FastText.
    Returns 0.0–1.0 score. Higher = more likely Russian.
    """
    if not text.strip():
        return 0.0
    try:
        pred = ft_model.predict(re.sub(r'\s+', ' ', text.strip()), k=1)
        label = pred[0][0].replace("__label__", "")
        conf = float(pred[1][0])
        return conf if label == "ru" else 0.0
    except Exception:
        return 0.0


def select_best_transcription(text_ru: str, text_uz: str) -> Tuple[str, LanguageResult]:
    """
    Select the best STT transcription from RU and UZ candidates.

    Approach: score-based comparison using FastText on BOTH transcriptions.
    - UZ score = how Uzbek/Turkic the UZ transcription looks
    - RU score = how Russian the RU transcription looks
    - UZ gets a +0.25 bias (Uzbek-first app)
    - Winner takes all

    Returns (best_text, language_result).
    """
    text_ru = (text_ru or "").strip()
    text_uz = (text_uz or "").strip()

    # Trivial cases
    if not text_ru and not text_uz:
        return "", LanguageResult("UZ_LATN", 0.0, 3)
    if text_uz and not text_ru:
        return text_uz, detect_language(text_uz)
    if text_ru and not text_uz:
        return text_ru, detect_language(text_ru)

    # Both have text — score each side
    print(f"🔤 [STT SELECT] RU='{text_ru}' | UZ='{text_uz}'")

    _ensure_model()

    if _ft_model is not None:
        try:
            uz_score = _uzbek_score(text_uz, _ft_model)
            ru_score = _russian_score(text_ru, _ft_model)

            print(f"🔤 [STT SELECT] uz_score={uz_score:.2f} | ru_score={ru_score:.2f}")

            # KEY INSIGHT from real data:
            # - Uzbek speech → UZ STT produces Latin text → uz_score > 0 (usually)
            # - Russian speech → UZ STT produces garbled Latin (sv/id/sr) → uz_score == 0
            #
            # EDGE CASE: Uzbek with many loanwords (davlat, ekspertiza) → FastText confused
            #   → uz_score == 0, but text is still Latin script = Uzbek
            #   → only pick RU if ru_score is HIGH (≥0.80), not just any Russian confidence

            uz_is_latin = not _has_cyrillic(text_uz)
            ru_is_cyrillic = _has_cyrillic(text_ru)

            if uz_score > 0:
                # Clear Turkic signal in UZ transcription → definitely Uzbek
                lang_code = "UZ_CYRL" if not uz_is_latin else "UZ_LATN"
                print(f"🔤 [STT SELECT] → UZ (Turkic signal={uz_score:.2f})")
                return text_uz, LanguageResult(lang_code, min(uz_score + 0.25, 1.0), 2)

            # uz_score == 0: no clear Turkic signal
            # Check for uniquely Uzbek grammar words before picking RU
            # These words (nima, uchun, kerak, bilan...) NEVER appear in Russian/Serbian
            _UZ_FUNCTION_WORDS = r'\b(nima|uchun|kerak|bilan|haqida|agar|lekin|yoki|ham|emas|bormi|yo\'q|ha|yo|bu|shu|u|men|sen|biz|siz)\b'
            if uz_is_latin and re.search(_UZ_FUNCTION_WORDS, text_uz.lower()):
                print(f"🔤 [STT SELECT] → UZ (function-word match in uz_score=0 case)")
                return text_uz, LanguageResult("UZ_LATN", 0.65, 2)

            # Use script + high-confidence RU as the deciding factor
            if ru_is_cyrillic and ru_score >= 0.90:
                # RU text is very strongly Russian → speaker likely spoke Russian
                print(f"🔤 [STT SELECT] → RU (uz_score=0, strong RU={ru_score:.2f})")
                return text_ru, LanguageResult("RU", ru_score, 2)

            # uz_score==0 but RU confidence not high enough → UZ default
            # (handles loanword-heavy Uzbek like "davlat ekologik ekspertizasi")
            lang_code = "UZ_CYRL" if not uz_is_latin else "UZ_LATN"
            print(f"🔤 [STT SELECT] → UZ (default: uz_score=0 but ru_score={ru_score:.2f}<0.90)")
            return text_uz, LanguageResult(lang_code, 0.5, 3)

        except Exception as e:
            print(f"⚠️ [STT SELECT] FastText error: {e}")

    # Fallback: UZ default
    lang = detect_language(text_uz)
    if lang.lang == "RU" and not _has_cyrillic(text_uz):
        lang = LanguageResult("UZ_LATN", 0.5, 3)
    return text_uz, lang


# Pre-load model on import
_ensure_model()
