import os
import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .settings import Settings
from .language_detector import detect_language, select_best_transcription, LanguageResult

settings = Settings()

# Query expansion: convert ordinal words to Roman numerals for better FAISS matching
ORDINAL_TO_ROMAN = {
    # Uzbek Latin
    "birinchi": "I",
    "ikkinchi": "II", 
    "uchinchi": "III",
    "to'rtinchi": "IV",
    "tortinchi": "IV",
    "beshinchi": "V",
    # Russian
    "первая": "I",
    "первой": "I", 
    "первую": "I",
    "первый": "I",
    "вторая": "II",
    "второй": "II",
    "вторую": "II",
    "третья": "III",
    "третьей": "III",
    "третий": "III",
    "четвертая": "IV",
    "четвертой": "IV",
    "четвертый": "IV",
    # Category keywords
    "kategoriya": "toifa",
    "категория": "toifa",
    "категории": "toifa",
    "категорию": "toifa",
}

# Short-query auto-expansion: when user types just 1-2 words, expand to a full
# searchable phrase so FAISS can find the right documents
SHORT_QUERY_EXPANSIONS = {
    # category / toifa queries
    "kategoriya": "541-sonli qaror toifalar kategoriyalar ro'yxati I II III IV ekologik ekspertiza",
    "kategoriyalar": "541-sonli qaror toifalar kategoriyalar ro'yxati I II III IV ekologik ekspertiza",
    "toifa": "541-sonli qaror toifalar kategoriyalar ro'yxati I II III IV ekologik ekspertiza",
    "toifalar": "541-sonli qaror toifalar kategoriyalar ro'yxati I II III IV ekologik ekspertiza",
    "категория": "541 постановление категории I II III IV экологическая экспертиза виды деятельности",
    "категории": "541 постановление категории I II III IV экологическая экспертиза виды деятельности",
    "категорию": "541 постановление категории I II III IV экологическая экспертиза виды деятельности",
    # specific categories
    "1 kategoriya": "I toifa birinchi kategoriya 541-sonli qaror ekologik ekspertiza",
    "2 kategoriya": "II toifa ikkinchi kategoriya 541-sonli qaror ekologik ekspertiza",
    "3 kategoriya": "III toifa uchinchi kategoriya 541-sonli qaror ekologik ekspertiza",
    "4 kategoriya": "IV toifa to'rtinchi kategoriya 541-sonli qaror ekologik ekspertiza",
    "1 toifa": "I toifa birinchi kategoriya 541-sonli qaror ekologik ekspertiza",
    "2 toifa": "II toifa ikkinchi kategoriya 541-sonli qaror ekologik ekspertiza",
    "3 toifa": "III toifa uchinchi kategoriya 541-sonli qaror ekologik ekspertiza",
    "4 toifa": "IV toifa to'rtinchi kategoriya 541-sonli qaror ekologik ekspertiza",
    "1 категория": "I категория первая экологическая экспертиза 541 постановление",
    "2 категория": "II категория вторая экологическая экспертиза 541 постановление",
    "3 категория": "III категория третья экологическая экспертиза 541 постановление",
    "4 категория": "IV категория четвёртая экологическая экспертиза 541 постановление",
    # ilova (appendix) queries
    "ilova": "541-sonli qaror ilova ro'yxat faoliyat turlari ekologik ekspertiza",
    "ilovalar": "541-sonli qaror ilova ro'yxat faoliyat turlari ekologik ekspertiza",
    "1 ilova": "1-ilova birinchi ilova 541-sonli qaror faoliyat turlari",
    "2 ilova": "2-ilova ikkinchi ilova 541-sonli qaror faoliyat turlari",
    "приложение": "541 постановление приложение виды деятельности категории экологическая экспертиза",
}

def num_to_uzbek_ordinal(n: int) -> str:
    """Dynamically converts any integer (up to 999) to its Uzbek ordinal string.
    Example: 14 -> o'n to'rtinchi, 20 -> yigirmanchi, 135 -> yuz o'ttiz beshinchi."""
    if n <= 0 or n > 999:
        return ""
        
    ones = {1: "bir", 2: "ikki", 3: "uch", 4: "to'rt", 5: "besh",
            6: "olti", 7: "yetti", 8: "sakkiz", 9: "to'qqiz"}
    tens = {10: "o'n", 20: "yigirma", 30: "o'ttiz", 40: "qirq", 50: "ellik",
            60: "oltmish", 70: "yetmish", 80: "sakson", 90: "to'qson"}
    
    ord_suffixes = {
        "bir": "inchi", "ikki": "nchi", "uch": "inchi", "to'rt": "inchi", "besh": "inchi",
        "olti": "nchi", "yetti": "nchi", "sakkiz": "inchi", "to'qqiz": "inchi",
        "o'n": "inchi", "yigirma": "nchi", "o'ttiz": "inchi", "qirq": "inchi", "ellik": "inchi",
        "oltmish": "inchi", "yetmish": "inchi", "sakson": "inchi", "to'qson": "inchi", "yuz": "inchi"
    }

    words = []
    
    h = n // 100
    if h > 0:
        if h == 1:
            words.append("yuz")
        else:
            words.append(f"{ones[h]} yuz")
            
    rem = n % 100
    t = (rem // 10) * 10
    o = rem % 10
    
    if t > 0:
        words.append(tens[t])
    if o > 0:
        words.append(ones[o])
        
    if not words:
        return ""
        
    last_word = words[-1].split()[-1] # handle "ikki yuz" -> "yuz"
    suffix = ord_suffixes.get(last_word, "inchi")
    
    words[-1] = words[-1] + suffix
    return " ".join(words)

def expand_query(query: str) -> str:
    """Expand query by adding Roman numeral equivalents for ordinal words
    and enhancing article/modda queries for better FAISS recall."""
    import re as _re
    query_lower = query.lower().strip()
    
    # For short queries: check if we have a direct expansion
    word_count = len(query_lower.split())
    if word_count <= 2 and query_lower in SHORT_QUERY_EXPANSIONS:
        expanded = SHORT_QUERY_EXPANSIONS[query_lower]
        print(f"🔍 [QUERY EXPANSION] Short query '{query}' -> '{expanded}'")
        return expanded
    
    expanded = query
    additions = []

    # Handle voice recognition errors ("1030 oltinchi" = 1036)
    query_lower = query_lower.replace("1030 oltinchi", "1036")
    query_lower = query_lower.replace("ming o'ttiz oltinchi", "1036")
    query_lower = query_lower.replace("ming o'ttiz olti", "1036")
    query_lower = query_lower.replace("bir ming o'ttiz oltinchi", "1036")
    query_lower = query_lower.replace("ming utiz oltinchi", "1036")

    # Expand article/modda queries: "20-modda" -> "20-modda yigirmanchi modda qonun bob"
    modda_match = _re.search(r'(\d+)[-\s]*(modda|модда|статья)', query_lower)
    if modda_match:
        num = int(modda_match.group(1))
        # Universally generate ordinal string for ANY article number
        ordinal = num_to_uzbek_ordinal(num)
        if ordinal:
            additions.append(f"{ordinal} modda")
        additions.append(f"{num}-modda qonun nizom qaror moddasi moddasida")

    for word, roman in ORDINAL_TO_ROMAN.items():
        if word in query_lower:
            if roman not in ["toifa"]:
                additions.append(f"{roman} toifa")
    
    if additions:
        expanded = query + " " + " ".join(additions)
        print(f"🔍 [QUERY EXPANSION] '{query}' -> '{expanded}'")
    
    return expanded

# Get the project root directory (parent of 'app' folder)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def ensure_file_exists(path):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("VM 541-sonli qarori matni.")


# Localized responses for non-RAG intents (handled by system prompt guard)
GREETING_RESPONSES = {
    "RU": "Здравствуйте! Чем могу помочь по вопросам экологической экспертизы?",
    "UZ_LATN": "Assalomu alaykum! Ekologik ekspertiza bo'yicha nima yordam bera olaman?",
    "UZ_CYRL": "Ассалому алайкум! Экологик экспертиза бўйича нима ёрдам бера оламан?"
}

IDENTITY_RESPONSES = {
    "RU": "Я — ИИ-помощник Центра государственной экологической экспертизы.",
    "UZ_LATN": "Men Davlat ekologik ekspertizasi markazining sun'iy intellekt yordamchisiman.",
    "UZ_CYRL": "Мен Давлат экологик экспертизаси марказининг сунъий интеллект ёрдамчисиман."
}

THANKS_RESPONSES = {
    "RU": "Пожалуйста! Если будут ещё вопросы по экологической экспертизе — обращайтесь.",
    "UZ_LATN": "Arzimaydi! Ekologik ekspertiza bo'yicha yana savollar bo'lsa — yozing.",
    "UZ_CYRL": "Арзимайди! Экологик экспертиза бўйича яна саволлар бўлса — ёзинг."
}

FAREWELL_RESPONSES = {
    "RU": "До свидания! Обращайтесь, если понадобится помощь.",
    "UZ_LATN": "Xayr! Yordam kerak bo'lsa, yozing.",
    "UZ_CYRL": "Хайр! Ёрдам керак бўлса, ёзинг."
}

# Fast intent patterns — ONLY match if the ENTIRE query is short (greeting-like)
# or the pattern appears as a standalone word
import re as _re

GREETING_PATTERNS = [
    # (regex pattern, max_query_word_count or None for any)
    (r'^salom$', None),
    (r'^salom\b', 5),  # "salom, qanday" but not inside longer queries
    (r'^assalomu\s+alaykum', None),
    (r'^привет$', None),
    (r'^привет\b', 5),
    (r'^здравствуйте', None),
    (r'^здравствуй$', None),
    (r'^privet$', None),
    (r'^zdravstvuyte', None),
    (r'^hello$', None),
    (r'^салом$', None),
    (r'^салом\b', 5),
]

IDENTITY_PATTERNS = [
    r'\bsen\s+kimsan\b',
    r'\bisming\s+nima\b',
    r'\bкто\s+ты\b',
    r'\bты\s+кто\b',
    r'\bkim\s+san\b',
    r'\bwho\s+are\s+you\b',
    r'\bsiz\s+kimsiz\b',
    r'\bсен\s+кимсан\b',
    r'\bисминг\s+нима\b',
]

# Gratitude patterns — catch "спасибо", "рахмат", "thanks", etc.
THANKS_PATTERNS = [
    (r'\bспасибо\b', 6),
    (r'\bблагодар', 6),
    (r'\bрахмат\b', 6),
    (r'\brahmat\b', 6),
    (r'\bthanks\b', 6),
    (r'\bthank\s+you\b', 6),
    (r'\bташаккур\b', 6),
    (r'\btashakkur\b', 6),
    (r'\bкатта рахмат\b', None),
    (r'\bkatta rahmat\b', None),
]

# Farewell patterns — catch "до свидания", "пока", "хайр", etc.
FAREWELL_PATTERNS = [
    (r'^пока$', None),
    (r'\bдо свидания\b', 5),
    (r'^хайр$', None),
    (r'^xayr$', None),
    (r'\bбуваринг\b', 5),
    (r'^bye$', None),
    (r'^goodbye$', None),
]


# Knowledge base files — all legal documents for RAG
KNOWLEDGE_BASE_FILES = [
    "541.txt",     # VM 541-sonli qarori (Постановление КМ №541)
    "1036.txt",    # Ekologik ekspertiza qonuni (Закон об экологической экспертизе, 2024)
    "14.txt",      # Ekologik normativlar nizomi (Порядок экологических нормативов)
    "1.txt",       # Faoliyat turlari ro'yxati — ilova (Приложение — список видов деятельности)
]

class BrainSystem:
    def __init__(self, file_paths=None):
        # Use default knowledge base files if not specified
        if file_paths is None:
            file_paths = KNOWLEDGE_BASE_FILES
        
        # Resolve file paths relative to project root for Docker compatibility
        self.file_paths = []
        for fp in file_paths:
            if not os.path.isabs(fp):
                self.file_paths.append(str(PROJECT_ROOT / fp))
            else:
                self.file_paths.append(fp)
        
        ensure_file_exists(self.file_paths[0]) # ensure at least 541 exists
        self.vector_store = None
        self.init_error = None
        self.chat_history = []
        self._initialize()

    def _initialize(self):
        print(f"🧠 [BRAIN] Загрузка системы (Vertex AI RAG v4.0 — Multi-doc + Flash + FastText)...")
        print(f"📁 [BRAIN] Файлы базы знаний: {len(self.file_paths)}")
        for fp in self.file_paths:
            exists = os.path.exists(fp)
            symbol = '✅' if exists else '❌'
            print(f"  {symbol} {os.path.basename(fp)}")
        
        if not settings.google_cloud_project and not settings.google_api_key and not settings.vertex_ai_api_key:
            self.init_error = "GOOGLE_CLOUD_PROJECT не настроен"
            print(f"❌ [BRAIN] ERROR: {self.init_error}")
            return
        
        os.environ["GOOGLE_CLOUD_PROJECT"] = settings.google_cloud_project
        os.environ["GOOGLE_CLOUD_LOCATION"] = settings.google_cloud_location
        if settings.vertex_ai_api_key:
            os.environ["GOOGLE_API_KEY"] = settings.vertex_ai_api_key
        
        print(f"☁️ [BRAIN] Vertex AI Project: {settings.google_cloud_project}")
        print(f"🌍 [BRAIN] Region: {settings.google_cloud_location}")

        try:
            # Load all knowledge base files
            all_documents = []
            for fp in self.file_paths:
                if not os.path.exists(fp):
                    print(f"⚠️ [BRAIN] Файл не найден, пропускаем: {fp}")
                    continue
                
                if fp.endswith('.pdf'):
                    loader = PyPDFLoader(fp)
                else:
                    loader = TextLoader(fp, encoding='utf-8', autodetect_encoding=True)
                
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source_file'] = os.path.basename(fp)
                all_documents.extend(docs)
                print(f"📄 [BRAIN] {os.path.basename(fp)}: загружено {len(docs)} документов")
            
            if not all_documents:
                self.init_error = "Нет доступных файлов базы знаний"
                print(f"❌ [BRAIN] ERROR: {self.init_error}")
                return
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
            chunks = text_splitter.split_documents(all_documents)
            print(f"📄 [BRAIN] Создано чанков: {len(chunks)}")

            api_key = settings.google_api_key or settings.vertex_ai_api_key

            if api_key:
                print("🔐 [BRAIN] Provider: Google GenAI API key")
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=api_key
                )
                self.vector_store = FAISS.from_documents(chunks, embeddings)

                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    api_key=api_key,
                    temperature=0.1,
                    max_tokens=8192,
                )
            else:
                # Vertex AI embedding publisher models are served from regional endpoints.
                embedding_location = (
                    "us-central1"
                    if settings.google_cloud_location == "global"
                    else settings.google_cloud_location
                )
                print(f"🔎 [BRAIN] Embedding region: {embedding_location}")
                print("🔐 [BRAIN] Provider: Vertex AI")

                embeddings = VertexAIEmbeddings(
                    model_name="gemini-embedding-001",
                    project=settings.google_cloud_project,
                    location=embedding_location
                )
                self.vector_store = FAISS.from_documents(chunks, embeddings)
                
                self.llm = ChatVertexAI(
                    model="gemini-2.5-flash",
                    temperature=0.1,
                    max_tokens=8192,
                    project=settings.google_cloud_project,
                    location=settings.google_cloud_location
                )
            
            print("✅ [BRAIN] Система готова (Multi-doc + Flash + FastText mode)!")

        except Exception as e:
            self.init_error = str(e)
            print(f"❌ [BRAIN] ERROR: {e}")

    def select_best_transcription(self, candidates: dict) -> str:
        """Select the best transcription using FastText ML (replaces LLM call)."""
        text_ru = candidates.get("ru", "").strip()
        text_uz = candidates.get("uz", "").strip()
        
        best_text, lang_result = select_best_transcription(text_ru, text_uz)
        print(f"🎤 [BRAIN] Selected '{best_text}' (lang={lang_result})")
        
        # Store detected language for later use
        self._detected_lang = lang_result
        return best_text

    def _detect_simple_intent(self, query: str, lang: str):
        """Fast pattern matching for greetings, identity, thanks, farewell (no LLM needed)."""
        query_lower = query.lower().strip()
        word_count = len(query_lower.split())
        
        # Helper to match pattern lists with optional max_words guard
        def _match_patterns(patterns, responses):
            for item in patterns:
                if isinstance(item, tuple):
                    pattern, max_words = item
                    if max_words and word_count > max_words:
                        continue
                    if _re.search(pattern, query_lower):
                        return responses.get(lang, responses["UZ_LATN"])
                else:
                    if _re.search(item, query_lower):
                        return responses.get(lang, responses["UZ_LATN"])
            return None
        
        # Check greeting patterns
        result = _match_patterns(GREETING_PATTERNS, GREETING_RESPONSES)
        if result:
            return result
        
        # Check identity patterns (no max_words, wrap in tuples)
        for pattern in IDENTITY_PATTERNS:
            if _re.search(pattern, query_lower):
                return IDENTITY_RESPONSES.get(lang, IDENTITY_RESPONSES["UZ_LATN"])
        
        # Check gratitude patterns
        result = _match_patterns(THANKS_PATTERNS, THANKS_RESPONSES)
        if result:
            return result
        
        # Check farewell patterns
        result = _match_patterns(FAREWELL_PATTERNS, FAREWELL_RESPONSES)
        if result:
            return result
        
        return None  # Not a simple intent — proceed to RAG

    def get_answer(self, query: str) -> str:
        start_total = time.time()
        
        if not self.vector_store:
            if self.init_error:
                return f"Baza yuklanmadi: {self.init_error}"
            return "Baza yuklanmadi. Loglarni tekshiring."

        # 1. Detect language via FastText (instant, <1ms)
        t1 = time.time()
        # Use cached detection from select_best_transcription if available
        if hasattr(self, '_detected_lang') and self._detected_lang:
            lang_result = self._detected_lang
            self._detected_lang = None  # Clear cache
        else:
            lang_result = detect_language(query)
        
        lang = lang_result.lang
        t2 = time.time()
        print(f"⏱️ Language detection (FastText): {(t2-t1)*1000:.1f}ms -> {lang} ({lang_result.confidence:.2f})")

        # 2. Check for simple intents (greeting, identity) — no LLM needed
        simple_response = self._detect_simple_intent(query, lang)
        if simple_response:
            print(f"⏱️ Simple intent handled in {(time.time()-start_total)*1000:.0f}ms")
            self._record_metrics(query, simple_response, "SIMPLE", lang, "none", 0, 0)
            return simple_response

        # 3. Build FAISS search query (augment short follow-ups with history context)
        search_query = query
        if len(query.split()) <= 5 and self.chat_history:
            # Short follow-up: prepend last user message for better context
            last_user = next(
                (m['content'] for m in reversed(self.chat_history) if m['role'] == 'user'),
                None
            )
            if last_user:
                search_query = f"{last_user} {query}"
        search_query = expand_query(search_query)
        docs = self.vector_store.similarity_search(search_query, k=25)
        context_text = "\n\n".join([d.page_content for d in docs])

        # 4. Build language-specific FULL prompt
        if lang == "RU":
            offtopic_note = "КРИТИЧЕСКИ ВАЖНО: Если вопрос НЕ является вопросом об экологии, строительстве, экспертизе, категориях объектов, постановлениях, нормативах или законах об экологической экспертизе — ОБЯЗАТЕЛЬНО ответь: 'Kechirasiz, men bu savolga javob bera olmayman. Men Vazirlar Mahkamasining 541 hamda 1036-qarorlariga muvofiq javob beraman. Hohlasangiz operatorlar bilan bog\\'laning: 📞 Tel: +998 71 203 03 04' НЕ используй контекст из базы знаний для ответа на нерелевантные вопросы. Примеры нерелевантных вопросов: благодарности, приветствия, личные вопросы, общие знания, шутки, математика."
            rag_prompt = f"""Ты — эксперт по экологическому законодательству Республики Узбекистан (Постановление КМ №541, Закон об экологической экспертизе №1036, нормативы экологической экспертизы).

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context_text}

ВОПРОС: {query}

ПРАВИЛА ОТВЕТА:
1. ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ (кириллица). Никакого узбекского текста.
2. Отвечай только на основе контекста выше.
3. ОТВЕЧАЙ ТОЧНО НА ВОПРОС:
   - Если спрашивают "что входит?", "перечисли" — дай полный список.
   - Если спрашивают "какая категория?" — назови категорию и коротко объясни. НЕ перечисляй весь список.
   - Если спрашивают о конкретном объекте — укажи только его категорию.
4. {offtopic_note}

ПРАВИЛА ФОРМАТИРОВАНИЯ:
- Для списков используй цифры: 1. 2. 3.
- Каждый пункт с новой строки.
- Не повторяй вопрос.
- Используй арабские цифры (1, 2, 3...), не римские.

ОТВЕТ:
"""
        else:
            offtopic_note_uz = "MUHIM: Agar savol ekologiya, qurilish, ekspertiza, obyektlar toifasi, qonunlar yoki normativ hujjatlarga ALOQADOR BO'LMASA — ALBATTA javob ber: 'Kechirasiz, men bu savolga javob bera olmayman. Men Vazirlar Mahkamasining 541 hamda 1036-qarorlariga muvofiq javob beraman. Hohlasangiz operatorlar bilan bog\\'laning: 📞 Tel: +998 71 203 03 04' Kontekstdagi ma'lumotlarni ALOQASIZ savollarga javob berish uchun ISHLATMA."
            if lang == "UZ_CYRL":
                offtopic_note_uz = "Агар савол олдинги суҳбатнинг давоми ёки аниқлаштириш бўлса ва контекстдан тушуниш мумкин бўлса — нормал жавоб бер. Агар савол ЯҚҚОЛ экология, қурилиш, экспертиза ёки норматившларга АЛОҚАДОР БЎЛМАСА — жавоб бер: 'Kechirasiz, men bu savolga javob bera olmayman. Men Vazirlar Mahkamasining 541 hamda 1036-qarorlariga muvofiq javob beraman. Hohlasangiz operatorlar bilan bog\\'laning: 📞 Tel: +998 71 203 03 04'"
            lang_instruction = "JAVOBNI FAQAT O'ZBEK TILIDA (LOTIN ALIFBOSIDA) BER." if lang != "UZ_CYRL" else "ЖАВОБНИ ФАҚАТ ЎЗБЕК ТИЛИДА (КИРИЛЛ АЛИФБОСИДА) БЕР."
            rag_prompt = f"""Sen O'zbekiston Respublikasining ekologik qonunchiligi bo'yicha ekspertsan (VM 541-sonli qaror, Ekologik ekspertiza to'g'risidagi qonun, ekologik normativlar).

KONTEKST:
{context_text}

SAVOL: {query}

JAVOB QOIDALARI:
1. {lang_instruction}
2. Faqat kontekstdagi ma'lumotlar asosida javob ber.
3. SAVOLGA ANIQ JAVOB BER:
   - Agar "nimalar kiradi?", "ro'yxat", "sanab ber" so'ralsa — to'liq ro'yxat keltir.
   - Agar "bu qaysi toifa?", "qaysi kategoriya?" so'ralsa — faqat tegishli toifani ayt va qisqacha tushuntir. Butun ro'yxatni BERMA.
   - Agar aniq biror faoliyat haqida so'ralsa — faqat o'sha faoliyat va uning toifasini ayt.
4. {offtopic_note_uz}

FORMATLASH QOIDALARI:
- Ro'yxat uchun raqamlar ishlat: 1. 2. 3.
- Har bir band YANGI QATORDAN boshlansin.
- Rim raqamlari ishlatma, faqat arab raqamlari.

JAVOB:
"""
        
        # 5. Single LLM call
        t3 = time.time()
        response = self.llm.invoke(rag_prompt)
        t4 = time.time()
        print(f"⏱️ LLM (Flash): {(t4-t3)*1000:.0f}ms")
        print(f"⏱️ TOTAL: {(t4-start_total)*1000:.0f}ms")
        
        result = response.content
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": result})
        
        tokens_in = len(rag_prompt) // 4
        tokens_out = len(result) // 4
        self._record_metrics(query, result, "ECOLOGY", lang, "gemini-2.5-flash", tokens_in, tokens_out)
        
        return result

    def get_answer_stream(self, query: str):
        """
        Streaming version of get_answer(). 
        Yields text chunks as they're generated by the LLM.
        Returns full answer via .full_answer attribute after iteration.
        """
        if not self.vector_store:
            error_msg = f"Baza yuklanmadi: {self.init_error}" if self.init_error else "Baza yuklanmadi."
            yield error_msg
            return

        # 1. Language detection (instant)
        if hasattr(self, '_detected_lang') and self._detected_lang:
            lang_result = self._detected_lang
            self._detected_lang = None
        else:
            lang_result = detect_language(query)
        lang = lang_result.lang

        # 2. Simple intent check
        simple_response = self._detect_simple_intent(query, lang)
        if simple_response:
            self._record_metrics(query, simple_response, "SIMPLE", lang, "none", 0, 0)
            yield simple_response
            return

        # 3. FAISS search (augment short follow-ups with history)
        search_query = query
        if len(query.split()) <= 5 and self.chat_history:
            last_user = next(
                (m['content'] for m in reversed(self.chat_history) if m['role'] == 'user'),
                None
            )
            if last_user:
                search_query = f"{last_user} {query}"
                print(f"[RAG STREAM] Context-augmented search for short query")
        search_query = expand_query(search_query)
        docs = self.vector_store.similarity_search(search_query, k=25)
        context_text = "\n\n".join([d.page_content for d in docs])

        # 4. Build language-specific full prompt
        if lang == "RU":
            offtopic_note = "КРИТИЧЕСКИ ВАЖНО: Если вопрос НЕ является вопросом об экологии, строительстве, экспертизе, категориях объектов, постановлениях, нормативах или законах об экологической экспертизе — ОБЯЗАТЕЛЬНО ответь: 'Извините, я эксперт только по экологическому законодательству Республики Узбекистан.' НЕ используй контекст из базы знаний для ответа на нерелевантные вопросы."
            rag_prompt = f"""Ты — эксперт по экологическому законодательству Республики Узбекистан (Постановление КМ №541, Закон об экологической экспертизе №1036, нормативы экологической экспертизы).

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context_text}

ВОПРОС: {query}

ПРАВИЛА ОТВЕТА:
1. ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ (кириллица). Никакого узбекского текста.
2. Отвечай только на основе контекста выше.
3. ОТВЕЧАЙ ТОЧНО НА ВОПРОС:
   - Если спрашивают что входит, перечисли — дай полный список.
   - Если спрашивают какая категория — назови категорию и уровень риска. НЕ пытайся объяснять причину, почему класс именно такой (особенно не связывай это со сроками экспертизы). НЕ перечисляй весь список.
   - Если спрашивают о конкретном объекте — укажи только его категорию и риск.
4. {offtopic_note}

ПРАВИЛА ФОРМАТИРОВАНИЯ:
- Для списков используй цифры: 1. 2. 3.
- Каждый пункт с новой строки.
- Используй арабские цифры (1, 2, 3...), не римские.

ОТВЕТ:
"""
        else:
            lang_instruction = "JAVOBNI FAQAT O'ZBEK TILIDA (LOTIN ALIFBOSIDA) BER." if lang != "UZ_CYRL" else "ЖАВОБНИ ФАҚАТ ЎЗБЕК ТИЛИДА (КИРИЛЛ АЛИФБОСИДА) БЕР."
            offtopic_note_uz = "MUHIM: Agar savol ekologiya, qurilish, ekspertiza, obyektlar toifasi, qonunlar yoki normativ hujjatlarga ALOQADOR BO'LMASA — ALBATTA javob ber: 'Kechirasiz, men bu savolga javob bera olmayman. Men Vazirlar Mahkamasining 541 hamda 1036-qarorlariga muvofiq javob beraman. Hohlasangiz operatorlar bilan bog\\'laning: 📞 Tel: +998 71 203 03 04' Kontekstdagi ma'lumotlarni ALOQASIZ savollarga javob berish uchun ISHLATMA."
            if lang == "UZ_CYRL":
                offtopic_note_uz = "Агар савол олдинги суҳбатнинг давоми ёки аниқлаштириш бўлса ва контекстдан тушуниш мумкин бўлса — нормал жавоб бер. Агар савол ЯҚҚОЛ экология, қурилиш, экспертиза ёки норматившларга АЛОҚАДОР БЎЛМАСА — жавоб бер: 'Kechirasiz, men bu savolga javob bera olmayman. Men Vazirlar Mahkamasining 541 hamda 1036-qarorlariga muvofiq javob beraman. Hohlasangiz operatorlar bilan bog\\'laning: 📞 Tel: +998 71 203 03 04'"

            rag_prompt = f"""Sen O'zbekiston Respublikasining ekologik qonunchiligi bo'yicha ekspertsan (VM 541-sonli qaror, Ekologik ekspertiza to'g'risidagi qonun, ekologik normativlar).

KONTEKST:
{context_text}

SAVOL: {query}

JAVOB QOIDALARI:
1. {lang_instruction}
2. Faqat kontekstdagi ma'lumotlar asosida javob ber.
3. SAVOLGA ANIQ JAVOB BER:
   - Agar nimalar kiradi, sanab ber so'ralsa — to'liq ro'yxat keltir.
   - Agar qaysi toifa so'ralsa — faqat tegishli toifani va xavf darajasini ayt. Nima uchun shu toifaga kirishiga sabablar o'ylab topma (masalan, ekspertiza kunlari sabab qilib ko'rsatilmasin). Butun ro'yxatni BERMA.
   - Agar aniq biror faoliyat haqida so'ralsa — faqat o'sha faoliyat, toifasi va xavf darajasini ayt.
4. {offtopic_note_uz}

FORMATLASH QOIDALARI:
- Ro'yxat uchun raqamlar ishlat: 1. 2. 3.
- Har bir band YANGI QATORDAN boshlansin.
- Rim raqamlari ishlatma, faqat arab raqamlari.

JAVOB:
"""

        # 5. Stream LLM response
        full_answer = ""
        t_start = time.time()

        for chunk in self.llm.stream(rag_prompt):
            text_chunk = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if text_chunk:
                full_answer += text_chunk
                yield text_chunk

        t_end = time.time()
        print(f"[LLM Stream] {(t_end-t_start)*1000:.0f}ms total")

        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": full_answer})

        tokens_in = len(rag_prompt) // 4
        tokens_out = len(full_answer) // 4
        self._record_metrics(query, full_answer, "ECOLOGY", lang, "gemini-2.5-flash", tokens_in, tokens_out)

    def _record_metrics(self, query: str, response: str, intent: str, lang: str, model: str, tokens_in: int, tokens_out: int):
        """Record metrics for analytics and save for logging."""
        try:
            from .metrics import metrics, CostCalculator
            cost = CostCalculator.calculate_gemini_cost(model, tokens_in, tokens_out)
            
            self._last_intent = intent
            self._last_language = lang
            self._last_model = model
            self._last_tokens_in = tokens_in
            self._last_tokens_out = tokens_out
            self._last_cost = cost
            
            print(f"📊 [RAG] {model}: {tokens_in}+{tokens_out} tokens, ${cost:.6f}")
        except Exception as e:
            print(f"⚠️ [METRICS] Error: {e}")

rag_system = BrainSystem()
