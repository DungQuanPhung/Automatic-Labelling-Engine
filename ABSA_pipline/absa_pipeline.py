import pandas as pd
from tqdm import tqdm

from text_processing import translate_to_english
from term_extraction import (
    split_and_term_extraction, 
    extract_terms_only_from_sentence,
    extract_terms_only_from_sentence_phi
)
from opinion_extraction import extract_opinions_only_from_clauses
from category_classifier import extract_category
from polarity_detector import detect_polarity


def absa_pipeline(
    sentences,
    model_llm,
    tokenizer_llm,
    model_cat,
    tokenizer_cat,
    id2label,
    device,
    save_every=100,
    save_path="/content/drive/MyDrive/ABSA_results/absa_results",
    use_split=False,
    use_phi=False
):
    """
    ABSA Pipeline đầy đủ
    
    Args:
        sentences: List of sentences hoặc single sentence
        model_llm: LLM model (Qwen hoặc Phi-3)
        tokenizer_llm: LLM tokenizer
        model_cat: Category classification model
        tokenizer_cat: Category tokenizer
        id2label: Label mapping cho category
        device: Device (cuda/cpu)
        save_every: Số clauses để auto-save
        save_path: Path để lưu kết quả
        use_split: True để split câu thành clauses, False để không split
        use_phi: True nếu dùng Phi-3, False nếu dùng Qwen
    """
    if isinstance(sentences, str):
        sentences = [sentences]

    all_clauses = []
    total_clauses = 0
    batch_index = 1

    for i, sentence in enumerate(tqdm(sentences, desc="Processing ABSA"), start=1):
        # Step 0: Translate nếu cần
        sentence = translate_to_english(sentence, model_llm, tokenizer_llm)

        # Step 1: Extract terms (và split nếu cần)
        if use_split:
            clauses = split_and_term_extraction(sentence, model_llm, tokenizer_llm)
        else:
            if use_phi:
                clauses = extract_terms_only_from_sentence_phi(sentence, model_llm, tokenizer_llm)
            else:
                clauses = extract_terms_only_from_sentence(sentence, model_llm, tokenizer_llm)

        # Step 2: Extract opinions
        clauses = extract_opinions_only_from_clauses(clauses, model_llm, tokenizer_llm)

        # Gắn chỉ số
        for c in clauses:
            c["sentence_index"] = i
            c["sentence_original"] = sentence

        all_clauses.extend(clauses)
        total_clauses += len(clauses)

        # Auto-save
        if total_clauses >= save_every:
            print(f"\nĐã trích xuất {total_clauses} clauses — đang xử lý Category & Polarity...")

            part_clauses = extract_category(all_clauses, model_cat, tokenizer_cat, id2label)
            part_clauses = detect_polarity(part_clauses)

            df_temp = pd.DataFrame(part_clauses)
            cols = [
                "sentence_index", "sentence_original",
                "clause", "term", "opinion", "category", "polarity", "polarity_score"
            ]
            df_temp = df_temp[[c for c in cols if c in df_temp.columns]]

            filename = f"{save_path}_part{batch_index}.csv"
            df_temp.to_csv(filename, index=False, encoding="utf-8-sig")
            print(f"✅ Lưu thành công: {filename}")

            all_clauses = []
            total_clauses = 0
            batch_index += 1

    # Xử lý phần còn lại
    if all_clauses:
        print(f"\nXử lý {len(all_clauses)} clauses cuối cùng ...")
        all_clauses = extract_category(all_clauses, model_cat, tokenizer_cat, id2label)
        all_clauses = detect_polarity(all_clauses)

        df_final = pd.DataFrame(all_clauses)
        cols = [
            "sentence_index", "sentence_original",
            "clause", "term", "opinion", "category", "polarity", "polarity_score"
        ]
        df_final = df_final[[c for c in cols if c in df_final.columns]]

        filename = f"{save_path}_final.csv"
        df_final.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"🎉 Hoàn thành toàn bộ! Kết quả lưu tại: {filename}")


def absa_pipeline_batch_save(
    all_sentences,
    model,
    tokenizer,
    max_new_tokens_term=20,
    max_new_tokens_opinion=40,
    batch_size=100,
    use_phi=False,
    output_prefix="absa_results"
):
    """
    ABSA pipeline đơn giản không dùng category classifier,
    tự động lưu CSV theo batch
    
    Args:
        all_sentences: List of sentences
        model: LLM model
        tokenizer: LLM tokenizer
        max_new_tokens_term: Max tokens cho term extraction
        max_new_tokens_opinion: Max tokens cho opinion extraction
        batch_size: Số sentences mỗi batch
        use_phi: True nếu dùng Phi-3
        output_prefix: Prefix cho output files
    """

    rows_buffer = []
    file_index = 1

    iterator = tqdm(
        enumerate(all_sentences),
        total=len(all_sentences),
        desc="📊 ABSA Processing"
    )

    for idx, sentence in iterator:
        # Step 1: Extract terms
        try:
            if use_phi:
                clauses = extract_terms_only_from_sentence_phi(
                    sentence, model, tokenizer, max_new_tokens=max_new_tokens_term
                )
            else:
                clauses = extract_terms_only_from_sentence(
                    sentence, model, tokenizer, max_new_tokens=max_new_tokens_term
                )
        except Exception:
            continue

        if not clauses:
            continue

        # Step 2: Extract opinions
        try:
            clauses_with_opinion = extract_opinions_only_from_clauses(
                clauses, model, tokenizer, max_new_tokens=max_new_tokens_opinion
            )
        except Exception:
            continue

        # Collect rows
        for c in clauses_with_opinion:
            rows_buffer.append({
                "sentence": sentence,
                "clause": c.get("clause", ""),
                "term": c.get("term", ""),
                "opinion": c.get("opinion", "")
            })

        # Save every batch_size
        if (idx + 1) % batch_size == 0:
            df = pd.DataFrame(rows_buffer)
            output_path = f"{output_prefix}_part_{file_index}.csv"
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"\n✅ Saved: {output_path} | Rows: {len(df)}")

            rows_buffer = []
            file_index += 1

    # Save remaining
    if rows_buffer:
        df = pd.DataFrame(rows_buffer)
        output_path = f"{output_prefix}_part_{file_index}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ Saved & downloaded FINAL: {output_path} | Rows: {len(df)}")

    print("🎉 ALL DONE")