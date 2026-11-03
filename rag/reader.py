import platform
import re

# The document reader: an open instruction-tuned LLM that writes a short answer from the
# retrieved chunks. Default is Qwen2.5-7B-Instruct (Sept 2024, 7.6B params), run with
# MLX (4-bit) on Apple Silicon or with HF transformers everywhere else.

DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Answer styles: "short" (v1) always answers with a short phrase. "adaptive" (v2) answers
# descriptive questions ("what is the significance of...", "what is X known for?") with one
# full sentence, since their reference answers are sentences and F1 is token overlap.
# "v3" = adaptive, plus: questions asking for a bare value (price, year, how many...) never get
# sentence mode; the short prompt asks for complete dates/names/units and for numbers that belong
# to the subject asked about; refusals and malformed answers are re-asked once.
# "v4" = v3 with a sentence prompt matched to the references' style (~15-20 words, the question's own
# wording, a general description without extra numbers/dates/names), and if the re-ask still refuses,
# a last answer from the model's own knowledge (closed-book) instead of keeping the refusal.
ANSWER_STYLES = ["short", "adaptive", "v3", "v4"]
V3_STYLES = ("v3", "v4")  # styles with the value guard, the v3 short prompt and the re-ask
DESCRIBE = re.compile(r"\b(significance|significant|specialty|specialties|unique|known for|famous for|tradition|"
                      r"distinction|role|describe|explain|why|how (does|do)|, and how|what (type|kind)s? of|"
                      r"experience)\b")
YES_NO = re.compile(r"^(is|are|was|were|does|do|did|can|could|should|has|have)\b")
DEFINE = re.compile(r"^what (is|are|was|were) (the )?['\"‘“]")  # what is the 'Kiltie Band'...?
# v3: questions asking for a bare value stay short even if a DESCRIBE word appears
# ("What is the price range for a VIP experience...?"). Not who/which: on dev, "Which X is
# known for Y?" questions have full-sentence references.
VALUE_QUESTION = re.compile(r"^(how (many|much|long|tall|high|old|far|big)|what (year|time|date|day|month)|"
                            r"what (is|was|are|were) the (price|cost|address|phone|length|height|capacity|age)s?)\b")
# replies that decline to answer (always score 0)
REFUSAL = re.compile(r"\b(unknown|not (specified|mentioned|provided|available|given|stated|listed|included)|"
                     r"does(n't| not) (provide|specify|mention|contain|say|include|state|list)|"
                     r"no (information|mention|details)|cannot (be )?determined?|n/a)\b", re.I)


def wants_sentence(question, value_guard=False):
    """Rule-based router for the adaptive style: True = answer with a full sentence."""
    q = question.lower().strip()
    if value_guard and VALUE_QUESTION.match(q):
        return False
    short_definition = re.match(r"^what (is|are|was|were) ", q) and len(q.split()) <= 5  # what is X?
    return bool(DESCRIBE.search(q) or YES_NO.match(q) or DEFINE.match(q) or short_definition)


def malformed(answer, sentence):
    """A short answer that is really a list or a dump (e.g. budget account codes)."""
    return not sentence and (len(answer.split()) > 15 or len(re.findall(r"\b\d{4,}\b", answer)) >= 3)

SYSTEM_RAG = (
    "You answer factual questions about Pittsburgh and Carnegie Mellon University (CMU) "
    "using the provided context documents. Reply with ONLY the answer: a short phrase, name, "
    "number or date, with no explanation and no full sentence. Copy names, numbers and "
    "titles exactly as written in the context. If the context does not contain the answer, "
    "give your best short guess. Never reply N/A, unknown, or that the context lacks the answer."
)
# v3 additions to the short prompt
SYSTEM_RAG_V3 = SYSTEM_RAG + (
    " Give the answer in full as the context writes it: a complete date (month, day and year) when "
    "the question asks for a date, a person's full name, and units or currency (e.g. '250 feet', "
    "'$3 million'). Only use a number if the document states it for the exact subject the question "
    "asks about, not for a different place or item."
)
# v3: second try after a refusal or malformed answer
REASK = ("That is not a usable answer. Answer the question again with your single best guess: {form}. "
         "Use the documents if they help; otherwise use what you know. Never say the information is "
         "unknown or not in the documents.")
SYSTEM_CLOSED = (
    "You answer factual questions about Pittsburgh and Carnegie Mellon University (CMU). "
    "Reply with ONLY the answer: a short phrase, name, number or date, with no explanation "
    "and no full sentence. If you are unsure, give your best short guess. Never reply N/A or unknown."
)
SYSTEM_SENTENCE = (
    "You answer questions about Pittsburgh and Carnegie Mellon University (CMU) using the provided "
    "context documents. Reply with ONE sentence of about 12-25 words that states the answer directly, "
    "the way an encyclopedia would. Start with the subject itself (or 'In <year>,' when the question "
    "is about a year, or 'Yes,' / 'No,' for a yes/no question), then give the key facts, reusing the "
    "context's own words, names and numbers. Do not restate the question: never open with 'The "
    "significance of ... is that', '... is significant because', or 'Visitors can find'. No lists and "
    "no second sentence. If the context does not contain the answer, still give your best one-sentence "
    "answer. Never reply N/A, unknown, or that the context does not say."
)
# format examples taken from the assignment handout
FEWSHOT = [
    ("Who is Pittsburgh named after?", "William Pitt"),
    ("What famous machine learning venue had its first conference in Pittsburgh in 1980?", "ICML"),
]
# sentence-style format examples (topics checked to not appear in the dev, leaderboard or test sets)
FEWSHOT_SENTENCE = [
    ("What is Primanti Bros. known for?",
     "Primanti Bros. is known for its sandwiches piled with french fries and coleslaw, served in "
     "Pittsburgh's Strip District since 1933."),
    ("What is the significance of the Carnegie Library of Pittsburgh?",
     "The Carnegie Library of Pittsburgh is the city's public library system, founded in 1895 with "
     "funding from Andrew Carnegie."),
    ("What is Mr. Yuk?",
     "Mr. Yuk is a green frowning-face poison warning symbol created at Children's Hospital of "
     "Pittsburgh in 1971."),
    ("What is the significance of the year 1816 for Pittsburgh?",
     "In 1816, Pittsburgh was incorporated as a city."),
    ("Is the Allegheny Observatory part of the University of Pittsburgh?",
     "Yes, the Allegheny Observatory is an astronomical research facility of the University of "
     "Pittsburgh's Department of Physics and Astronomy."),
]
# v4: dev references for descriptive questions are short, general descriptions that reuse the question's
# wording; v3's answers were often correct but worded differently and padded with specific numbers,
# dates and names the references don't have, which costs token-overlap F1
SYSTEM_SENTENCE_V4 = (
    "You answer questions about Pittsburgh and Carnegie Mellon University (CMU) using the provided "
    "context documents. Reply with ONE sentence of about 15-20 words, written like a short travel-guide "
    "or encyclopedia description. Start with the subject, named with the same words the question uses "
    "(or 'In <year>,' when the question is about a year, or 'Yes,' / 'No,' for a yes/no question). "
    "Reuse the question's key phrase (e.g. 'is known for', 'specializes in', 'is significant as'), then "
    "say in general terms what the subject is and what it is known for. Leave out extra numbers, dates, "
    "counts and people's names unless the question asks for them. No lists and no second sentence. If the "
    "context does not contain the answer, still give your best one-sentence answer from what you know. "
    "Never reply N/A, unknown, or that the context does not say."
)
# same topics as FEWSHOT_SENTENCE (checked absent from the eval sets), minus details nobody asked for
FEWSHOT_SENTENCE_V4 = [
    ("What is Primanti Bros. known for?",
     "Primanti Bros. is known for its sandwiches piled high with french fries and coleslaw."),
    ("What is the significance of the Carnegie Library of Pittsburgh?",
     "The Carnegie Library of Pittsburgh is significant as the city's public library system, founded with "
     "funding from Andrew Carnegie."),
    ("What is Mr. Yuk?",
     "Mr. Yuk is a green frowning-face symbol created in Pittsburgh to warn children away from poisons."),
    ("What is the significance of the year 1816 for Pittsburgh?",
     "In 1816, Pittsburgh was incorporated as a city."),
    ("Is the Allegheny Observatory part of the University of Pittsburgh?",
     "Yes, the Allegheny Observatory is an astronomical research facility of the University of Pittsburgh."),
]
# v4: last resort after a refused re-ask, answering from the model's own knowledge
SYSTEM_CLOSED_SENTENCE = (
    "You answer questions about Pittsburgh and Carnegie Mellon University (CMU) from your own knowledge. "
    "Reply with ONE sentence of about 15-20 words that starts with the subject and says what it is or what "
    "it is known for. If you are unsure, give your best guess. Never reply N/A, unknown, or that you do not know."
)


def build_messages(question, contexts=None, sentence=False, v3=False, v4=False):
    if contexts is None:
        system, shots = (SYSTEM_CLOSED_SENTENCE, FEWSHOT_SENTENCE_V4) if sentence else (SYSTEM_CLOSED, FEWSHOT)
        msgs = [{"role": "system", "content": system}]
        for q, a in shots:
            msgs += [{"role": "user", "content": f"Question: {q}"}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": f"Question: {question}"})
        return msgs
    ctx = "\n\n".join(f"[Document {i}] {c}" for i, c in enumerate(contexts, 1))
    if sentence:
        system, shots = (SYSTEM_SENTENCE_V4, FEWSHOT_SENTENCE_V4) if v4 else (SYSTEM_SENTENCE, FEWSHOT_SENTENCE)
    else:
        system, shots = (SYSTEM_RAG_V3 if v3 else SYSTEM_RAG), FEWSHOT
    msgs = [{"role": "system", "content": system}]
    for q, a in shots:
        msgs += [{"role": "user", "content": f"Question: {q}"}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"})
    return msgs


def clean_answer(text):
    text = text.strip().split("\n")[0].strip()
    for prefix in ("Answer:", "A:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.strip().strip('"').rstrip(".").strip()


class Reader:
    def __init__(self, model=None, backend=None, max_new_tokens=48, answer_style="short", sentence_tokens=80):
        if backend is None:
            backend = "mlx" if platform.system() == "Darwin" and platform.machine() == "arm64" else "hf"
        self.backend, self.max_new_tokens = backend, max_new_tokens
        self.answer_style, self.sentence_tokens = answer_style, sentence_tokens
        if backend == "mlx":
            from mlx_lm import load
            self.model, self.tokenizer = load(model or DEFAULT_MLX_MODEL)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            name = model or DEFAULT_HF_MODEL
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")

    def generate(self, messages, max_tokens=None):
        """Greedy-decode a reply to a list of chat messages."""
        max_tokens = max_tokens or self.max_new_tokens
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.backend == "mlx":
            from mlx_lm import generate
            return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return self.tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def sentence_mode(self, question, contexts=None):
        """Whether this question gets a full-sentence answer (adaptive/v3/v4 style, RAG only)."""
        return (self.answer_style in ("adaptive", *V3_STYLES) and contexts is not None
                and wants_sentence(question, value_guard=self.answer_style in V3_STYLES))

    def answer(self, question, contexts=None):
        sentence = self.sentence_mode(question, contexts)
        v3, v4 = self.answer_style in V3_STYLES, self.answer_style == "v4"
        max_tokens = self.sentence_tokens if sentence else None
        messages = build_messages(question, contexts, sentence, v3, v4)
        raw = self.generate(messages, max_tokens)
        answer = clean_answer(raw)
        self.retried = self.fallback = False
        if v3 and (REFUSAL.search(answer) or malformed(answer, sentence)):
            form = "one sentence" if sentence else "a short name, number or date"
            messages += [{"role": "assistant", "content": raw.strip()},
                         {"role": "user", "content": REASK.format(form=form)}]
            answer, self.retried = clean_answer(self.generate(messages, max_tokens)), True
            if v4 and contexts is not None and REFUSAL.search(answer):
                # still refusing: a guess from the model's own knowledge can score, a refusal never does
                answer = clean_answer(self.generate(build_messages(question, None, sentence), max_tokens))
                self.fallback = True
        return answer
