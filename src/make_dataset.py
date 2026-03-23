from src.preprocess_funcs import clean_data, load_raw_data, save_processed_data


def main():
    
    df = load_raw_data()
    df_clean = clean_data(df)
    save_processed_data(df_clean)


if __name__ == "__main__":
    main()