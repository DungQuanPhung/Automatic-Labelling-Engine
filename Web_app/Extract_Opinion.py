import re
from Extract_Clause_Term import chat_auto
from config import MAX_NEW_TOKENS_OPINION, OPINION_BATCH_SIZE

def make_batch_prompt_en(batch):
    prompt = (
        "You are an expert in Aspect-Based Sentiment Analysis (ABSA).\n\n"
        "Task:\n"
        "For each clause below, extract ALL opinion expressions (adjectives, adverbial phrases, evaluative words/phrases, noun/verb phrases, comparative, superlative, negation, quantity, outcome, result, etc.) that directly describe or evaluate the given term/aspect.\n"
        "Only extract opinion words/phrases that appear EXACTLY in the clause.\n"
        "Keep negations, intensifiers, and quantity words attached. Do NOT paraphrase, translate, or invent.\n"
        "If there is NO opinion, leave the answer BLANK (do NOT write 'None', 'N/A', or any label).\n\n"
        
        "Strict rules:\n"
        "1. Extract only opinions that clearly describe or evaluate the main term.\n"
        "2. Include adjectives that describe the term (e.g., 'clean', 'friendly', 'comfortable').\n"
        "3. Include adverb + adjective combinations (e.g., 'very helpful', 'extremely rude').\n"
        "4. Include verb phrases that express sentiment (e.g., 'easy to relax', 'hard to find').\n"
        "5. Include outcome/effect expressions that imply sentiment (e.g., 'smooth', 'enjoyable', 'worth the price').\n"
        "6. Include comparative/superlative expressions (e.g., 'better than expected', 'the best').\n"
        "7. Include negated opinions and absence expressions (e.g., 'no hot water', 'not helpful', 'lack of variety').\n"
        "8. Include intensifiers with opinions (e.g., 'too noisy', 'very close', 'at all').\n"
        "9. Include availability/variety/quantity phrases (e.g., 'variety of options', 'plenty of choices').\n"
        "10. If multiple opinion expressions exist, extract ALL of them in their original order.\n"
        "11. If clause describes results/outcomes caused by the term, treat those as opinions about the term.\n"
        "12. For coordinated opinions (e.g., 'clean and comfortable'), SEPARATE them into individual opinions: 'clean, comfortable'.\n"
        "13. REMOVE conjunctions like 'and', 'or', 'but' between opinions. Only keep the core opinion words.\n\n"
        
        "Special handling:\n"
        "- For fragments without subject/predicate but containing opinion-bearing words, extract those opinion words.\n"
        "- For continuation phrases (e.g., 'at all', 'in our room'), look for modifiers or opinion expressions.\n"
        "- For clauses describing features related to the term (e.g., 'Amazing view from balcony'), extract the descriptive part.\n"
        "- Include 'worth' expressions (e.g., 'totally worth the price', 'worth every penny').\n"
        "- For prepositional phrases with evaluative content (e.g., 'with great service'), extract the evaluative part.\n"
        "- Include exclamatory expressions that convey sentiment (e.g., 'Amazing!', 'Excellent!', 'Terrible!').\n"
        "- Include price-value judgments (e.g., 'overpriced', 'good value').\n\n"
        
        "Output format: NUMBER. opinion1, opinion2,...\n"
        "Do NOT add any explanation, label, or extra output.\n\n"
        "Data:\n"
    )
    for idx, item in enumerate(batch):
        prompt += f"{idx+1}. Term: '{item.get('term', '')}' | Clause: '{item['clause']}'\n"
    prompt += "\nAnswers:"
    return prompt

def clean_opinion(text: str) -> str:
    """Clean opinion text, remove unwanted tokens/words."""
    if not text:
        return ""
    # Remove numberings at the start (e.g., "1. ", "2. ", "10. ", etc.)
    text = re.sub(r'^\d+\.\s*', '', text)
    text = re.sub(r'^ID\s*', '', text, flags=re.IGNORECASE)
    text = text.rstrip('.')
    text = re.sub(r'<[^>]+>', '', text)  # Remove <...>
    text = re.sub(r'\([^)]+\)', '', text)  # Remove (...)
    
    # Loại bỏ phần "Answer:" nếu model lặp lại
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    
    # Loại bỏ các câu giải thích dài
    sentences = text.split(".")
    if len(sentences) > 1:
        text = sentences[0].strip()
    
    # Loại bỏ các từ khóa của instructions
    skip_phrases = ["extract", "opinion", "about", "from the clause", "the term", 
                   "describes", "evaluates", "there is no", "output format"]
    for phrase in skip_phrases:
        if phrase in text.lower() and len(text) > 30:
            text = ""
            break
    
    unwanted = ["opinion:", "answer:", "output:"]
    for word in unwanted:
        text = re.sub(rf'^{word}\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_opinions_only_from_clauses(
    clauses, model, tokenizer, max_new_tokens=MAX_NEW_TOKENS_OPINION, batch_size=OPINION_BATCH_SIZE, model_type="gguf"
):
    if not clauses:
        return clauses

    final_clauses = []

    for i in range(0, len(clauses), batch_size):
        batch = clauses[i:i + batch_size]
        prompt_text = make_batch_prompt_en(batch)
        messages = [{"role": "user", "content": prompt_text}]
        response = chat_auto(model, tokenizer, messages, max_new_tokens=max_new_tokens * batch_size, model_type=model_type)
        lines = response.strip().split('\n')
        for idx, item in enumerate(batch):
            opinion_extracted = ""
            prefix = f"{idx+1}."
            for line in lines:
                if line.strip().startswith(prefix):
                    # Nếu chỉ có số thứ tự, hoặc số thứ tự + dấu chấm, hoặc số thứ tự + khoảng trắng, thì coi là không có opinion
                    content = line.split(prefix, 1)[-1].strip()
                    if not content or content.lower() in ["none", "<none>", "n/a"]:
                        opinion_extracted = ""
                    else:
                        opinion_extracted = content
                    break
            # Làm sạch output
            opinion_extracted = clean_opinion(opinion_extracted)
            
            # Nếu vẫn còn "None" hoặc chỉ là số thứ tự, thì để trống
            if opinion_extracted.lower() in ["none", "<none>", "n/a"] or re.match(r'^\d+\.?$', opinion_extracted):
                opinion_extracted = ""
            
            # Chuẩn hóa danh sách opinions giống như file thường
            if opinion_extracted:
                opinions = re.split(r",", opinion_extracted)
                opinions = [o.strip() for o in opinions if o.strip()]
                
                # Lọc bỏ những "opinion" quá dài (có thể là explanation)
                opinions = [o for o in opinions if len(o.split()) <= 7]
                
                # Chỉ giữ opinions xuất hiện trong clause hoặc sentence gốc
                clause_text = item.get("clause", "")
                sentence_original = item.get("sentence_original", "")
                valid_opinions = []
                for o in opinions:
                    if re.search(rf"\b{re.escape(o)}\b", clause_text, re.IGNORECASE):
                        valid_opinions.append(o)
                    elif re.search(rf"\b{re.escape(o)}\b", sentence_original, re.IGNORECASE):
                        valid_opinions.append(o)
                
                opinion_extracted = ", ".join(valid_opinions) if valid_opinions else ""
            # Đảm bảo thứ tự key: clause, term, opinion, sentence_original
            new_c = {
                "clause": item.get("clause", ""),
                "term": item.get("term", ""),
                "opinion": opinion_extracted,
                "sentence_original": item.get("sentence_original", "")
            }
            final_clauses.append(new_c)

    return final_clauses