import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def load_and_preprocess_data(file_path, column_name, output_column='human_category'):
    """
    Load và preprocess một file data
    
    Args:
        file_path (str): Đường dẫn đến file
        column_name (str): Tên column cần lấy
        output_column (str): Tên column đầu ra
    
    Returns:
        pd.Series: Series chứa data đã preprocess
    """
    # Xác định file type
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Lấy column và đổi tên
    data = df[column_name]
    data.name = output_column
    
    # Lowercase
    data = data.str.lower()
    
    return data

def merge_multiple_annotators(data_list):
    """
    Merge data từ nhiều annotators
    
    Args:
        data_list (list): List of pd.Series từ các annotators
    
    Returns:
        pd.Series: Merged data
    """
    df_merged = pd.concat(data_list, axis=0, ignore_index=True)
    return df_merged

def calculate_agreement(annotator1_data, annotator2_data):
    """
    Tính simple agreement percentage
    
    Args:
        annotator1_data (pd.Series): Data từ annotator 1
        annotator2_data (pd.Series): Data từ annotator 2
    
    Returns:
        float: Agreement percentage
    """
    agreement = (annotator1_data == annotator2_data).mean()
    return agreement

def calculate_cohens_kappa(annotator1_data, annotator2_data, weights=None):
    """
    Tính Cohen's Kappa score
    
    Args:
        annotator1_data (pd.Series): Data từ annotator 1
        annotator2_data (pd.Series): Data từ annotator 2
        weights (str, optional): 'linear' hoặc 'quadratic' cho weighted kappa
    
    Returns:
        float: Cohen's Kappa score
    """
    kappa = cohen_kappa_score(
        annotator1_data.astype(str),
        annotator2_data.astype(str),
        weights=weights
    )
    return kappa

def save_merged_data(data, output_path):
    """
    Lưu merged data ra file CSV
    
    Args:
        data (pd.Series): Data cần lưu
        output_path (str): Đường dẫn output file
    """
    data.to_csv(output_path, index=False)
    print(f"✅ Saved data to: {output_path}")

def analyze_inter_annotator_agreement(
    annotator1_path,
    annotator2_path,
    column1='human_category',
    column2='human_category',
    verbose=True
):
    """
    Phân tích toàn diện IAA giữa 2 annotators
    
    Args:
        annotator1_path (str): Path đến file của annotator 1
        annotator2_path (str): Path đến file của annotator 2
        column1 (str): Column name trong file 1
        column2 (str): Column name trong file 2
        verbose (bool): In kết quả chi tiết
    
    Returns:
        dict: Dictionary chứa các metrics
    """
    # Load data
    if verbose:
        print("📂 Loading data from annotators...")
    
    df1 = pd.read_csv(annotator1_path)
    df2 = pd.read_csv(annotator2_path)
    
    data1 = df1[column1].astype(str).str.lower()
    data2 = df2[column2].astype(str).str.lower()
    
    # Calculate metrics
    if verbose:
        print("📊 Calculating agreement metrics...")
    
    agreement = calculate_agreement(data1, data2)
    kappa = calculate_cohens_kappa(data1, data2)
    weighted_kappa = calculate_cohens_kappa(data1, data2, weights='quadratic')
    
    results = {
        'agreement': agreement,
        'cohens_kappa': kappa,
        'weighted_kappa': weighted_kappa,
        'n_samples': len(data1)
    }
    
    # Print results
    if verbose:
        print("\n" + "="*50)
        print("📈 INTER-ANNOTATOR AGREEMENT RESULTS")
        print("="*50)
        print(f"Number of samples: {results['n_samples']}")
        print(f"Simple Agreement: {results['agreement']:.2%}")
        print(f"Cohen's Kappa: {results['cohens_kappa']:.4f}")
        print(f"Weighted Kappa (Quadratic): {results['weighted_kappa']:.4f}")
        print("="*50)
        
        # Interpretation
        print("\n💡 Kappa Interpretation:")
        kappa_val = results['cohens_kappa']
        if kappa_val < 0:
            print("  - Poor agreement (worse than chance)")
        elif kappa_val < 0.20:
            print("  - Slight agreement")
        elif kappa_val < 0.40:
            print("  - Fair agreement")
        elif kappa_val < 0.60:
            print("  - Moderate agreement")
        elif kappa_val < 0.80:
            print("  - Substantial agreement")
        else:
            print("  - Almost perfect agreement")
    
    return results

def analyze_confusion_matrix(annotator1_data, annotator2_data, labels=None):
    """
    Tạo confusion matrix giữa 2 annotators
    
    Args:
        annotator1_data (pd.Series): Data từ annotator 1
        annotator2_data (pd.Series): Data từ annotator 2
        labels (list, optional): List of labels
    
    Returns:
        pd.DataFrame: Confusion matrix
    """
    
    if labels is None:
        labels = sorted(set(annotator1_data) | set(annotator2_data))
    
    cm = confusion_matrix(annotator1_data, annotator2_data, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    return cm_df

def find_disagreements(
    annotator1_path,
    annotator2_path,
    column1='human_category',
    column2='human_category',
    output_path=None
):
    """
    Tìm các cases mà 2 annotators không đồng ý
    
    Args:
        annotator1_path (str): Path đến file của annotator 1
        annotator2_path (str): Path đến file của annotator 2
        column1 (str): Column name trong file 1
        column2 (str): Column name trong file 2
        output_path (str, optional): Path để lưu disagreements
    
    Returns:
        pd.DataFrame: DataFrame chứa các disagreements
    """
    df1 = pd.read_csv(annotator1_path)
    df2 = pd.read_csv(annotator2_path)
    
    # Merge
    df1['annotator1'] = df1[column1].astype(str).str.lower()
    df2['annotator2'] = df2[column2].astype(str).str.lower()
    
    df_merged = pd.concat([df1, df2['annotator2']], axis=1)
    
    # Find disagreements
    disagreements = df_merged[df_merged['annotator1'] != df_merged['annotator2']]
    
    print(f"\n🔍 Found {len(disagreements)} disagreements out of {len(df_merged)} samples")
    print(f"   ({len(disagreements)/len(df_merged)*100:.2f}% disagreement rate)")
    
    # Save if output path provided
    if output_path:
        disagreements.to_csv(output_path, index=False)
        print(f"✅ Saved disagreements to: {output_path}")
    
    return disagreements

def calculate_per_category_agreement(
    annotator1_path,
    annotator2_path,
    column1='human_category',
    column2='human_category'
):
    """
    Tính agreement cho từng category
    
    Args:
        annotator1_path (str): Path đến file của annotator 1
        annotator2_path (str): Path đến file của annotator 2
        column1 (str): Column name trong file 1
        column2 (str): Column name trong file 2
    
    Returns:
        pd.DataFrame: Agreement scores per category
    """
    df1 = pd.read_csv(annotator1_path)
    df2 = pd.read_csv(annotator2_path)
    
    data1 = df1[column1].astype(str).str.lower()
    data2 = df2[column2].astype(str).str.lower()
    
    categories = sorted(set(data1) | set(data2))
    
    results = []
    for cat in categories:
        # Find samples where either annotator labeled as this category
        mask = (data1 == cat) | (data2 == cat)
        
        if mask.sum() > 0:
            cat_agreement = (data1[mask] == data2[mask]).mean()
            results.append({
                'category': cat,
                'n_samples': mask.sum(),
                'agreement': cat_agreement
            })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('agreement', ascending=False)
    
    print("\n📊 Per-Category Agreement:")
    print(df_results.to_string(index=False))
    
    return df_results

# ===== MAIN WORKFLOW =====

def main_merge_annotators():
    """
    Workflow để merge data từ nhiều annotators
    """
    print("🔄 Merging data from multiple annotators...")
    
    # Load data from 3 sources
    df1 = load_and_preprocess_data('/Quan.csv', 'category')
    df2 = load_and_preprocess_data('/70% sample.csv', 'category')
    df3 = load_and_preprocess_data('/quân.xlsx', 'human_category')
    
    # Merge
    df_all = merge_multiple_annotators([df1, df2, df3])
    
    print(f"✅ Merged {df_all.shape[0]} samples from 3 sources")
    
    # Save
    save_merged_data(df_all, "merged_annotations.csv")
    
    return df_all

def main_analyze_two_annotators():
    """
    Workflow để phân tích agreement giữa 2 annotators
    """
    # Analyze agreement
    results = analyze_inter_annotator_agreement(
        annotator1_path='/Data/qa_tuyen.csv',
        annotator2_path='/Data/qa_quan.csv',
        column1='human_category',
        column2='human_category',
        verbose=True
    )
    
    # Find disagreements
    disagreements = find_disagreements(
        annotator1_path='/Data/qa_tuyen.csv',
        annotator2_path='/Data/qa_quan.csv',
        column1='human_category',
        column2='human_category',
        output_path='disagreements.csv'
    )
    
    # Per-category agreement
    per_cat_results = calculate_per_category_agreement(
        annotator1_path='/Data/qa_tuyen.csv',
        annotator2_path='/Data/qa_quan.csv',
        column1='human_category',
        column2='human_category'
    )
    
    return results, disagreements, per_cat_results

if __name__ == "__main__":
    # 1. Merge data từ nhiều annotators
    # main_merge_annotators()
    
    # 2. Analyze agreement giữa 2 annotators
    main_analyze_two_annotators()