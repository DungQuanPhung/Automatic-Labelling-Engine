import pandas as pd
from google.colab import drive

# Import các modules
from config import DEVICE
from model_loader import load_qwen_model, load_phi3_model
from category_classifier import train_category_classifier, extract_category
from absa_pipeline import absa_pipeline, absa_pipeline_batch_save
from evaluation import evaluate_absa, print_evaluation_results, analyze_errors


def main_train_category_model():
    """Training category classifier"""
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Load data
    goal_path = "/content/500 sample.csv"
    goal_df = pd.read_csv(goal_path)
    
    # Train model
    model_cat, tokenizer_cat, label2id, id2label = train_category_classifier(goal_df)
    
    return model_cat, tokenizer_cat, label2id, id2label


def main_run_pipeline_with_qwen():
    """Chạy ABSA pipeline với Qwen model"""
    # Mount Drive
    drive.mount('/content/drive')
    
    # Load models
    print("Loading Qwen model...")
    model_qwen, tokenizer_qwen = load_qwen_model()
    
    print("Loading category model...")
    model_cat, tokenizer_cat, label2id, id2label = main_train_category_model()
    
    # Load data
    df_eval = pd.read_csv('/content/evaluation.csv')
    all_sentences = df_eval["clause"].tolist()
    
    # Run pipeline
    absa_pipeline(
        sentences=all_sentences,
        model_llm=model_qwen,
        tokenizer_llm=tokenizer_qwen,
        model_cat=model_cat,
        tokenizer_cat=tokenizer_cat,
        id2label=id2label,
        device=DEVICE,
        save_every=100,
        save_path="/content/drive/MyDrive/ABSA_results/absa_results",
        use_split=False,  # Không split câu
        use_phi=False
    )


def main_run_pipeline_with_phi3():
    """Chạy ABSA pipeline với Phi-3 model"""
    # Mount Drive
    drive.mount('/content/drive')
    
    # Load models
    print("Loading Phi-3 model...")
    model_phi, tokenizer_phi = load_phi3_model()
    
    print("Loading category model...")
    model_cat, tokenizer_cat, label2id, id2label = main_train_category_model()
    
    # Load data
    df_eval = pd.read_csv('/content/evaluation.csv')
    all_sentences = df_eval["clause"].tolist()
    
    # Run pipeline
    absa_pipeline(
        sentences=all_sentences,
        model_llm=model_phi,
        tokenizer_llm=tokenizer_phi,
        model_cat=model_cat,
        tokenizer_cat=tokenizer_cat,
        id2label=id2label,
        device=DEVICE,
        save_every=100,
        save_path="/content/drive/MyDrive/ABSA_results/absa_results",
        use_split=False,
        use_phi=True
    )


def main_simple_pipeline():
    """Chạy pipeline đơn giản không dùng category classifier"""
    # Load model
    print("Loading Phi-3 model...")
    model, tokenizer = load_phi3_model()
    
    # Load data
    df_eval = pd.read_csv('/content/evaluation.csv')
    all_sentences = df_eval["clause"].tolist()[700:]  # Lấy từ index 700
    
    # Run simple pipeline
    absa_pipeline_batch_save(
        all_sentences=all_sentences,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens_term=20,
        max_new_tokens_opinion=40,
        batch_size=100,
        use_phi=True,
        output_prefix="absa_results"
    )


def main_evaluate_results():
    """Đánh giá kết quả"""
    # Load results
    predict = pd.read_csv("/content/predict.csv")
    target = pd.read_csv("/content/target.csv")
    
    # Merge
    df_merged = pd.concat([predict, target], axis=1)
    
    # Evaluate
    results = evaluate_absa(df_merged)
    non_disc_df, disc_df = print_evaluation_results(results)
    
    # Analyze errors
    errors = analyze_errors(df_merged)
    print("\n📊 Error Analysis:")
    for key, value in errors.items():
        if 'samples' not in key:
            print(f"{key}: {value}")
    
    return results, errors


def main_single_sentence_test():
    """Test với một câu đơn"""
    # Load model
    model, tokenizer = load_phi3_model()
    
    # Test sentence
    sentence = "The room was clean but the staff were not very helpful and the location was perfect."
    
    # Run pipeline
    from term_extraction import extract_terms_only_from_sentence_phi
    from opinion_extraction import extract_opinions_only_from_clauses
    
    print("Extracting terms...")
    clauses = extract_terms_only_from_sentence_phi(sentence, model, tokenizer)
    
    print("\nExtracting opinions...")
    clauses = extract_opinions_only_from_clauses(clauses, model, tokenizer)
    
    print("\nResults:")
    for c in clauses:
        print(f"Clause: {c['clause']}")
        print(f"Term: {c['term']}")
        print(f"Opinion: {c['opinion']}")
        print("-" * 50)


if __name__ == "__main__":
    # 1. Train category model
    # main_train_category_model()
    
    # 2. Chạy full pipeline với Qwen
    # main_run_pipeline_with_qwen()
    
    # 3. Chạy full pipeline với Phi-3
    # main_run_pipeline_with_phi3()
    
    # 4. Chạy simple pipeline
    # main_simple_pipeline()
    
    # 5. Đánh giá kết quả
    # main_evaluate_results()
    
    # 6. Test với 1 câu
    main_single_sentence_test()