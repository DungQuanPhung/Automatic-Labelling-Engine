import re
from model_loader import chat, chat_phi3


def extract_opinions_only_from_clauses(clauses, model, tokenizer, max_new_tokens=40):
    """
    Extract ONLY opinion expressions for each clause in hotel domain.
    Hỗ trợ cả Qwen và Phi-3.
    """
    final_clauses = []

    for c in clauses:
        clause_text = c["sentence_original"]
        term = c.get("term", "")
        sentence_original = c.get("sentence_original", "")

        prompt = (
        "You are an expert linguist working on Aspect-Based Sentiment Analysis (ABSA) "
        "in the domain of hotel and hospitality reviews.\n\n"

        "==================== DOMAIN KNOWLEDGE ====================\n"
        "Common opinion expressions:\n"
        "• Cleanliness: clean, dirty, spotless, dusty\n"
        "• Service attitude: friendly, rude, helpful, unprofessional\n"
        "• Comfort: comfortable, noisy, spacious, small\n"
        "• Food: delicious, cold, amazing, awful\n"
        "• Value: expensive, overpriced, worth it\n"
        "• Location: convenient, far away, perfect location\n"
        "• Overall: perfect, terrible, disappointing, fantastic\n\n"

        "==================== STRICT RULES ====================\n"
        "1️⃣ Must keep original wording only.\n"
        "2️⃣ Opinion words must appear exactly in the clause.\n"
        "3️⃣ Only extract evaluative expressions.\n"
        "4️⃣ Must describe or evaluate the Term:\n"
        f"     → Term: '{term}'\n"
        "5️⃣ Do NOT guess or invent opinions.\n"
        "6️⃣ If no clear opinion: return empty.\n"
        "7️⃣ Respond ONLY in required format.\n\n"

        "==================== OUTPUT FORMAT ====================\n"
        "Opinion: <opinion1, opinion2, ...>\n"
        "(comma separated)\n\n"

        "==================== RESPONSE INSTRUCTION ====================\n"
        "No extra comments. No explanations. Only answer.\n\n"

        f"Clause:\n{clause_text}"
        )

        # Kiểm tra model type để gọi đúng chat function
        model_type = type(model).__name__
        
        if "Phi" in str(model.config._name_or_path):
            opinion_text = chat_phi3(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens
            ).strip()
        else:
            messages = [{"role": "user", "content": prompt}]
            opinion_text = chat(
                model, 
                tokenizer, 
                messages,
                max_new_tokens=max_new_tokens
            ).strip()

        # Clean output
        opinion_text = (
            opinion_text.replace("Opinion:", "")
            .replace("<|im_end|>", "")
            .replace("\n", " ")
            .strip()
        )

        opinions = [
            o.strip() for o in re.split(r",", opinion_text)
            if o.strip()
        ]

        # Validate: must appear in original sentence
        valid_opinions = [
            o for o in opinions
            if re.search(
                rf"\b{re.escape(o)}\b",
                sentence_original,
                re.IGNORECASE
            )
        ]

        new_c = c.copy()
        new_c["opinion"] = ", ".join(valid_opinions) if valid_opinions else ""
        final_clauses.append(new_c)

    return final_clauses