"""Offline fallback helpers, used only when the language model cannot be reached (the normal path is LLM-driven): language detection, small talk, Hindi/Hinglish keyword expansion
and an extractive fallback answer, so the bot still replies helpfully when the Gemini API is unavailable."""
import re

CONTACT = "admissions@liet.in, +91 9821582662"

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Common Roman-script Hindi words; a question containing any of them is treated as Hinglish.
_HINGLISH = set(
    """kya kaise kaun kab kahan kahaan kitna kitni kitne kyun kyu kaisa hai hain hoga hogi honge tha thi
    mujhe mujhko mera meri mere hume humein hamara hamari aap aapka aapki tum tumhara kaise kaisa batao
    bataiye bataye bata batana chahiye chahta chahti karna karni karein kare karte sakta sakti sakte
    milega milegi milta milti lena lene dena dene ka ki ke ko se mein pe aur nahi nahin haan
    bhi toh liye wala wali wale abhi kuch koi jaldi pehle baad lagega lagti lagta
    padhai padhna naukri dakhila daakhila shulk sampark pata kaha kahaa kidhar kaisi ho hoon hu main
    nhi nai ni rha rhi rhe h muje mjhe kyaa kia chaiye smjha smjhao smjh krdo kro krna btao btana yrr yar kru karu bhaiya scene hora hua
    namaste namaskar shukriya dhanyavad dhanyawad alvida tata madad naam mai tera teri tumhe aapko dikhao samjhao btao bta bataao yaar bhai dost""".split()
)


def detect_language(text: str) -> str:
    """'hi' (Devanagari), 'hinglish' (Hindi in Roman letters) or 'en'."""
    if _DEVANAGARI.search(text):
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    return "hinglish" if any(w in _HINGLISH for w in words) else "en"


# ---------- small talk ----------

def _norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # "hiiiii" -> "hii"
    return " ".join(re.findall(r"[a-z0-9ऀ-ॿ']+", text))


_SMALLTALK = [
    ("greeting", r"(h+i+|h+e+l+o+|h+e+y+|hii+|hey there|yo|namaste|namaskar|pranam|ram ram|sat sri akal|good (morning|afternoon|evening)|"
                 r"gm|नमस्ते|नमस्कार|हेलो|हैलो|हाय|हलो|प्रणाम|सुप्रभात)( (there|ji|sir|mam|madam|bot|liet|assistant|bhai|dost|friend))?"),
    ("howareyou", r"(how are you|how r u|how are u|hows it going|wassup|whats up|sup|kaise ho|kaisi ho|kaise hain|kya haal( hai)?|"
                  r"kya hal( hai)?|kya chal raha hai|aap kaise hain|aap kaise ho|tum kaise ho|आप कैसे हैं|कैसे हो|क्या हाल है|क्या हाल)( (bhai|yaar|ji))?"),
    ("thanks", r"(thanks|thanx|thnx|thnks|tysm|thank you|thank u|thanku|thx|ty|many thanks|thanks a lot|shukriya|shukriyaa|dhanyavad|dhanyawad|dhanyvaad|"
               r"bahut shukriya|bahut dhanyavad|thanks bhai|धन्यवाद|शुक्रिया|थैंक्स|थैंक यू|बहुत धन्यवाद)( (a lot|so much|bhai|ji|sir))?"),
    ("bye", r"(bye|bye bye|goodbye|good bye|see you|see ya|tata|alvida|phir milenge|chalo bye|ok bye|good night|gn|बाय|अलविदा|फिर मिलेंगे|शुभ रात्रि)"),
    ("ok", r"(ok|okay|k|kk|cool|great|nice|fine|theek hai|thik hai|achha|accha|acha|badhiya|ठीक है|अच्छा|बढ़िया)( (hai|thanks|bhai))?"),
    ("who", r"(who are you|who r u|what are you|what is your name|whats your name|your name|tum kaun ho|aap kaun hain|aap kaun ho|"
            r"tumhara naam kya hai|aapka naam kya hai|naam kya hai|tum kya ho|aap kya hain|तुम कौन हो|आप कौन हैं|आपका नाम क्या है|तुम्हारा नाम क्या है)"),
    ("help", r"(help|help me|what can you do|what do you do|how can you help|kya kar sakte ho|kya kar sakte hain|tum kya kar sakte ho|"
             r"aap kya kar sakte hain|madad|madad karo|help karo|mujhe help chahiye|मदद|मदद करो|आप क्या कर सकते हैं|तुम क्या कर सकते हो)"),
]
_SMALLTALK = [(name, re.compile(rf"^(?:{pat})$")) for name, pat in _SMALLTALK]

_REPLIES = {
    "greeting": {
        "en": "Hello! Welcome to LIET. I can help with admissions, courses, fees, hostel, placements and more. What would you like to know?",
        "hi": "नमस्ते! LIET में आपका स्वागत है। मैं एडमिशन, कोर्स, फीस, हॉस्टल और प्लेसमेंट के बारे में बता सकता हूँ। आप क्या जानना चाहेंगे?",
        "hinglish": "Namaste! LIET mein aapka swagat hai. Main admission, courses, fees, hostel aur placements ke baare mein bata sakta hoon. Aap kya jaanna chahenge?",
    },
    "howareyou": {
        "en": "I'm doing great, thanks for asking! How can I help you with LIET today?",
        "hi": "मैं बिल्कुल ठीक हूँ, पूछने के लिए धन्यवाद! बताइए, LIET के बारे में मैं आपकी क्या मदद करूँ?",
        "hinglish": "Main bilkul badhiya hoon, poochne ke liye thanks! Bataiye, LIET ke baare mein main aapki kya help karun?",
    },
    "thanks": {
        "en": "You're welcome! Feel free to ask anything else about LIET.",
        "hi": "आपका स्वागत है! LIET के बारे में कुछ और पूछना हो तो बेझिझक पूछिए।",
        "hinglish": "Aapka swagat hai! LIET ke baare mein aur kuch poochna ho toh bina jhijhak poochiye.",
    },
    "bye": {
        "en": "Goodbye! All the best. Come back anytime if you have more questions about LIET.",
        "hi": "अलविदा! आपको शुभकामनाएँ। LIET के बारे में और सवाल हों तो कभी भी आइए।",
        "hinglish": "Bye! All the best. LIET ke baare mein aur sawal ho toh kabhi bhi aana.",
    },
    "ok": {
        "en": "Great! Let me know if you'd like to know about courses, fees, admissions, hostel or placements.",
        "hi": "बढ़िया! कोर्स, फीस, एडमिशन, हॉस्टल या प्लेसमेंट के बारे में जानना हो तो बताइए।",
        "hinglish": "Badhiya! Courses, fees, admission, hostel ya placements ke baare mein jaanna ho toh bataiye.",
    },
    "who": {
        "en": "I'm the LIET Assistant, the AI helper of Lloyd Institute of Engineering & Technology, Greater Noida. Ask me about admissions, courses, fees, hostel, placements, library and more.",
        "hi": "मैं LIET असिस्टेंट हूँ, लॉयड इंस्टीट्यूट ऑफ इंजीनियरिंग एंड टेक्नोलॉजी, ग्रेटर नोएडा का AI हेल्पर। आप मुझसे एडमिशन, कोर्स, फीस, हॉस्टल, प्लेसमेंट और लाइब्रेरी के बारे में पूछ सकते हैं।",
        "hinglish": "Main LIET Assistant hoon, Lloyd Institute of Engineering & Technology, Greater Noida ka AI helper. Aap mujhse admission, courses, fees, hostel, placements aur library ke baare mein pooch sakte hain.",
    },
    "help": {
        "en": "I can answer questions about LIET: courses and eligibility, admission process, fees and scholarships, hostel and facilities, placements, library, research and contact details. You can ask in English, Hindi or Hinglish.",
        "hi": "मैं LIET से जुड़े सवालों के जवाब दे सकता हूँ: कोर्स और पात्रता, एडमिशन प्रक्रिया, फीस और स्कॉलरशिप, हॉस्टल और सुविधाएँ, प्लेसमेंट, लाइब्रेरी, रिसर्च और संपर्क जानकारी। आप हिंदी, अंग्रेज़ी या हिंग्लिश में पूछ सकते हैं।",
        "hinglish": "Main LIET se jude sawaalon ke jawab de sakta hoon: courses aur eligibility, admission process, fees aur scholarships, hostel aur facilities, placements, library, research aur contact details. Aap English, Hindi ya Hinglish mein pooch sakte hain.",
    },
}


def smalltalk(question: str) -> str | None:
    """Instant reply for greetings, thanks, 'who are you', etc. None when it is a real question."""
    norm = re.sub(r"^(bhai|yaar|yar|yrr|bro|arey|are|oye|hey|ok|achha|acha) (?=\S)", "", _norm(question))
    if not norm or len(norm.split()) > 6:
        return None
    for name, pattern in _SMALLTALK:
        if pattern.match(norm):
            lang = detect_language(question)
            if lang == "en" and _DEVANAGARI.search(question):
                lang = "hi"
            return _REPLIES[name][lang]
    return None


# ---------- Hindi / Hinglish -> English keywords (for retrieval when the LLM is unavailable) ----------

_GLOSSARY = {
    "fee fees fee structure": "fees fee shulk shulak फीस फ़ीस शुल्क kharcha kharch खर्च cost kitna kitni kitne कितना कितनी कितने paisa पैसे",
    "hostel accommodation": "hostel hostal हॉस्टल हास्टल छात्रावास rehna रहना rahna room रूम pg",
    "placement job package companies recruiters": "placement placements प्लेसमेंट naukri नौकरी job jobs package salary सैलरी company companies कंपनी कंपनियाँ campus recruitment",
    "admission apply application process": "admission admissions daakhila dakhila दाखिला प्रवेश एडमिशन apply अप्लाई application aavedan आवेदन registration",
    "courses programs branches": "course courses कोर्स कोर्सेस पाठ्यक्रम branch branches ब्रांच stream program programme vibhag",
    "eligibility criteria": "eligibility patrata patrta yogyata पात्रता योग्यता criteria percentage marks cutoff",
    "scholarship": "scholarship scholarships स्कॉलरशिप छात्रवृत्ति chhatravritti discount",
    "library": "library lib लाइब्रेरी पुस्तकालय pustakalaya books kitab किताब",
    "contact phone email address": "contact sampark संपर्क number phone फोन email mail address pata पता kahan kahaan कहाँ कहां location",
    "transport bus": "bus transport बस परिवहन",
    "canteen mess food": "canteen cafeteria mess कैंटीन मेस khana खाना food",
    "timings": "timing timings time samay समय baje बजे",
    "faculty teachers": "faculty teacher teachers professor shikshak शिक्षक अध्यापक",
    "research innovation": "research innovation रिसर्च अनुसंधान incubation startup",
    "ragging": "ragging रैगिंग anti",
    "exam": "exam exams examination pariksha परीक्षा result परिणाम",
    "syllabus": "syllabus पाठ्यक्रम curriculum",
    "duration": "duration avadhi अवधि years saal साल",
    "documents required": "documents document dastavez दस्तावेज़ required zaroori जरूरी",
    "facilities campus": "facilities facility suvidha सुविधा campus कैंपस wifi lab labs sports gym",
    "AKTU university affiliation": "aktu affiliation university vishwavidyalaya विश्वविद्यालय approved aicte",
    "last date deadline": "last date deadline antim tareekh अंतिम तारीख",
}
# trigger word -> English expansion terms
_TRIGGERS: dict[str, str] = {}
for _en, _words in _GLOSSARY.items():
    for _w in _words.split():
        _TRIGGERS[_w.lower()] = _en


def expand_query(question: str) -> str:
    """Append English keywords for Hindi/Hinglish trigger words so BM25/embeddings can match English pages."""
    toks = re.findall(r"[ऀ-ॿ]+|[a-z0-9]+", question.lower())
    extra: list[str] = []
    for t in toks:
        hit = _TRIGGERS.get(t)
        if hit is None and _DEVANAGARI.search(t):  # Devanagari suffixes: फीस, फीसें
            hit = next((v for k, v in _TRIGGERS.items() if _DEVANAGARI.search(k) and len(k) > 2 and k in t), None)
        if hit and hit not in extra:
            extra.append(hit)
    return (question + " " + " ".join(extra)).strip() if extra else question


# ---------- fallback answer ----------

_INTRO = {
    "en": "Here is what I found on the LIET website:",
    "hi": "LIET की वेबसाइट पर मुझे यह जानकारी मिली (अंग्रेज़ी में):",
    "hinglish": "LIET website par mujhe ye information mili (English mein):",
}
_CONFIRM = {
    "en": f"Please confirm the latest details with the admissions office: {CONTACT}",
    "hi": f"ताज़ा जानकारी के लिए कृपया एडमिशन ऑफिस से पुष्टि करें: {CONTACT}",
    "hinglish": f"Latest details ke liye please admissions office se confirm kar lijiye: {CONTACT}",
}
_NOT_FOUND = {
    "en": f"I couldn't find that detail on the LIET website. You can ask me about courses, admissions, fees, hostel or placements, or contact the admissions office: {CONTACT}",
    "hi": f"यह जानकारी मुझे LIET की वेबसाइट पर नहीं मिली। आप मुझसे कोर्स, एडमिशन, फीस, हॉस्टल या प्लेसमेंट के बारे में पूछ सकते हैं, या एडमिशन ऑफिस से संपर्क करें: {CONTACT}",
    "hinglish": f"Ye detail mujhe LIET website par nahi mili. Aap mujhse courses, admission, fees, hostel ya placements ke baare mein pooch sakte hain, ya admissions office se baat karein: {CONTACT}",
}
_STOP = set("a an the of in on at to for is are was were be been do does did can could i my me we you your it its and or if "
            "what which who whom how when where why with by from that this as any there than then so not no".split())


def _words(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1}


def fallback_answer(question: str, sources: list[dict], max_sentences: int = 4) -> str:
    """Build a reply from the retrieved excerpts alone (used when the language model is unavailable)."""
    lang = detect_language(question)
    query = _words(expand_query(question))
    scored: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for rank, src in enumerate(sources):
        text = re.sub(r"Source: \S+", " ", src["text"])
        text = re.sub(r"\s+", " ", text)
        for sent in re.split(r"(?<=[.!?])\s+|\s[•·|]\s", text):
            sent = sent.strip(" -•|")
            sent = sent if len(sent) <= 300 else sent[:300].rsplit(" ", 1)[0] + "…"
            if len(sent) < 25 or sent in seen or _DEVANAGARI.search(sent) or sent[0].islower():
                continue
            overlap = len(query & _words(sent))
            if overlap >= 2 or (overlap >= 1 and len(query) == 1):
                seen.add(sent)
                scored.append((overlap, -rank, sent))
    if not scored:
        return _NOT_FOUND[lang]
    best = sorted(scored, reverse=True)[:max_sentences]
    order = {s: i for i, (_, _, s) in enumerate(scored)}  # keep original reading order
    bullets = "\n".join(f"- {s}" for _, _, s in sorted(best, key=lambda x: order[x[2]]))
    return f"{_INTRO[lang]}\n\n{bullets}\n\n{_CONFIRM[lang]}"
