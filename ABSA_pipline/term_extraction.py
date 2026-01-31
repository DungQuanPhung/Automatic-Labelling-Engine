import re
from model_loader import chat, chat_phi3

def split_sentence_with_terms_llm(sentence, model, tokenizer, max_new_tokens=300):
    """
    Tách một sentence thành các clause + extract term/aspect chỉ bằng LLM.
    Trả về list dict: [{"clause": ..., "term": ..., "sentence_original": ...}, ...]
    """
    prompt = (
    "You are an expert linguist working on Aspect-Based Sentiment Analysis (ABSA).\n"
    "Your task is to split the following review sentence into smaller clauses and identify the aspect/term discussed in each clause.\n\n"

    "==================== STRICT RULES ====================\n"
    "1️. DO NOT add, remove, translate, explain, or modify ANY words, symbols, or punctuation in the original sentence.\n"
    "   • Every clause must be a **continuous substring** of the original sentence.\n"
    "   • The output must cover **all parts of the sentence** — no content should be ignored or missing.\n"
    "2️. Only split the sentence where it makes sense semantically — typically around conjunctions ('and', 'but', 'while', 'although', etc.) "
    "or when the opinion changes.\n"
    "   •Do NOT split phrases that grammatically or logically belong to the same subject. "
    "   • If a descriptive phrase does not have a clear term in the sentence, keep it as a separate clause but leave Term blank."
    "3️. Keep the exact original wording and order in each clause. Do NOT reorder, paraphrase, or summarize.\n"
    "4️. Each clause must express a clear **opinion or evaluative meaning**, either explicit (e.g., 'dirty', 'perfect') or implicit "
    "(e.g., 'gave us many tips' implies helpfulness, 'helped us with departure' implies good service).\n"
    "5️. Do NOT separate adverbs (e.g., 'really', 'very', 'so', 'too', 'quite', 'extremely', 'absolutely', "
    "'rather', 'fairly', 'pretty', 'incredibly', 'particularly', 'deeply', 'highly') from the words they modify.\n"
    "6️. Keep negative or limiting words such as 'nothing', 'none', 'nobody', 'no one', 'nowhere', 'never', "
    "'hardly', 'barely', 'scarcely', 'without', 'no', 'not' **inside the same clause** — they must not be removed or separated.\n"
    "7️. Identify the **TERM** being discussed in each clause.\n"
    "   • TERM: the main aspect or entity being described (e.g., 'staff', 'room', 'hotel').\n"
    "   • If no clear term appears, leave it blank.\n"
    "8️. Avoid creating meaningless or redundant clauses.\n"
    "9️. If multiple terms appear in the same clause, separate them with commas.\n"
    "10️. If a clause refers to the same entity as a previous one but does not repeat it explicitly, "
    "**propagate the term from the previous clause**.\n\n"

    "==================== COVERAGE REQUIREMENT ====================\n"
    " Every part of the original sentence must appear in at least one clause.\n"
    " Do NOT skip, shorten, or drop any meaningful phrase, even if it lacks an explicit sentiment word.\n"
    " Clauses that describe actions, experiences, or behaviors with clear positive/negative implications "
    "must be included (e.g., 'gave us many tips', 'helped us with departure').\n\n"

    "==================== OUTPUT FORMAT ====================\n"
    "Clause: <clause text> | Term: <term1,term2,...>\n\n"

    "==================== EXAMPLES ====================\n"
    "Input: The apartment was fully furnished, great facilities, everything was cleaned and well prepared.\n"
    "Output:\n"
    "Clause: The apartment was fully furnished | Term: apartment\n"
    "Clause: great facilities | Term: facilities\n"
    "Clause: everything was cleaned and well prepared | Term: room,facility\n\n"

    "Input: diny was really helpful, he gave us many tips and helped us with departure.\n"
    "Output:\n"
    "Clause: diny was really helpful | Term: staff\n"
    "Clause: he gave us many tips | Term: staff\n"
    "Clause: helped us with departure | Term: staff\n\n"

    "Input: i can definitely recommend it!.\n"
    "Output:\n"
    "Clause: i can definitely recommend it! | Term: \n\n"

    "==================== RESPONSE INSTRUCTION ====================\n"
    "Respond ONLY with the clauses and terms exactly in the format shown above.\n"
    "Do NOT include any explanation, reasoning, or commentary.\n"
    "Do NOT include quotation marks, markdown, or extra text.\n\n"

    f"Now process this sentence WITHOUT changing any words:\n{sentence}"
    )

    messages = [{"role": "user", "content": prompt}]
    response = chat(model, tokenizer, messages, max_new_tokens=max_new_tokens).strip()

    result = []
    last_term = ""
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "| Term:" in line:
            clause_text, term = line.split("| Term:")
            clause_text = clause_text.replace("Clause:", "").strip()
            term = term.strip()
            if term == "":
                term = last_term
            else:
                last_term = term
        else:
            clause_text = line
            term = last_term
        result.append({"clause": clause_text, "term": term, "sentence_original": sentence})

    return result

def extract_terms_only_from_sentence(sentence, model, tokenizer, max_new_tokens=20):
    """
    Extract only TERMS (aspects) from whole sentence using LLM.
    Do NOT split sentence into clauses.
    """
    prompt = (
        "You are an expert linguist specializing in Aspect-Based Sentiment Analysis (ABSA).\n"
        "Your task: Identify the **aspect/term** that is being described or evaluated in the entire sentence below.\n\n"

        "==================== DOMAIN ====================\n"
        "Domain: HOTEL reviews\n\n"

        "==================== STRICT RULES ====================\n"
        "1. TERM must appear as an explicit entity/aspect in the sentence.\n"
        "2. Do NOT paraphrase, translate, or create new terms.\n"
        "3. Term must be a noun related to hotel domain (e.g., staff, room,rooms, service, location, facility)\n"
        "4. If multiple terms appear → separate them by commas.\n"
        "5. If no clear term appears → leave it blank.\n\n"

        "==================== OUTPUT FORMAT ====================\n"
        "Term: <term1,term2,...>\n\n"

        "==================== RESPONSE INSTRUCTION ====================\n"
        "Respond ONLY with the term list.\n"
        "Do NOT include: Clause, quotes, explanation, extra text.\n\n"

        f"Sentence:\n{sentence}\n\n"
        "Answer:"
    )

    messages = [{"role": "user", "content": prompt}]
    term_text = chat(model, tokenizer, messages, max_new_tokens=max_new_tokens).strip()

    term_text = (
        term_text.replace("<|im_end|>", "")
        .replace("Term:", "")
        .replace("\n", " ")
        .strip()
    )

    terms = re.split(r",", term_text)
    terms = [t.strip() for t in terms if t.strip()]

    valid_terms = [
        t for t in terms if re.search(rf"\b{re.escape(t)}\b", sentence, re.IGNORECASE)
    ]

    return [{
        "sentence_original": sentence,
        "term": ", ".join(valid_terms) if valid_terms else "",
        "clause": sentence
    }]

def extract_terms_only_from_sentence_phi(sentence, model, tokenizer, max_new_tokens=20):
    """
    Extract only TERMS (aspects) from whole sentence using Phi-3.
    """
    if isinstance(sentence, list):
        sentence = " ".join([str(s) for s in sentence if s])
    if not isinstance(sentence, str):
        sentence = str(sentence)

    sentence = sentence.strip()

    if not sentence:
        return [{
            "sentence_original": "",
            "term": "",
            "clause": ""
        }]

    prompt = (
        "You are an expert linguist specializing in Aspect-Based Sentiment Analysis (ABSA).\n"
        "Your task: Identify the **aspect/term** that is being described or evaluated in the entire sentence below.\n\n"

        "==================== DOMAIN ====================\n"
        "Domain: HOTEL reviews\n\n"

        "==================== STRICT RULES ====================\n"
        "1. TERM must appear as an explicit entity/aspect in the sentence.\n"
        "2. Do NOT paraphrase, translate, or create new terms.\n"
        "3. Term must be a noun related to hotel domain "
        "(e.g., staff, room, rooms, service, location, facility).\n"
        "4. If multiple terms appear → separate them by commas.\n"
        "5. If no clear term appears → leave it blank.\n\n"

        "==================== OUTPUT FORMAT ====================\n"
        "Term: <term1,term2,...>\n\n"

        "==================== RESPONSE INSTRUCTION ====================\n"
        "Respond ONLY with the term list.\n"
        "Do NOT include Clause, quotes, explanation, or extra text.\n\n"

        f"Sentence:\n{sentence}"
    )

    term_text = chat_phi3(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens
    ).strip()

    term_text = (
        term_text.replace("<|im_end|>", "")
        .replace("Term:", "")
        .replace("\n", " ")
        .strip()
    )

    terms = [t.strip() for t in term_text.split(",") if t.strip()]

    valid_terms = [
        t for t in terms
        if re.search(rf"\b{re.escape(t)}\b", sentence, re.IGNORECASE)
    ]

    return [{
        "sentence_original": sentence,
        "term": ", ".join(valid_terms) if valid_terms else "",
        "clause": sentence
    }]

def split_and_term_extraction(sentence, model, tokenizer):
    """Kết hợp split và term extraction với refinement"""
    clauses = split_sentence_with_terms_llm(sentence, model=model, tokenizer=tokenizer)
    if clauses:
        last_clause = clauses[-1]
        if "term" in last_clause:
            last_clause["term"] = last_clause["term"].replace("<|im_end|>", "").strip()
    
    for c in clauses:
        terms = [t.strip() for t in c.get("term", "").split(",") if t.strip()]
        terms = [t for t in terms if t.lower() in c["sentence_original"].lower()]
        c["term"] = ",".join(terms)
    
    return clauses