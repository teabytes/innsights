# this script preprocesses hotel booking data, handling missing values and ensuring data quality
import pandas as pd
import os


# preprocess the data and save the cleaned version
def preprocess_data(input_path: str, output_path: str):
    # check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input file not found: {input_path}")
    
    # load dataset
    df = pd.read_csv(input_path)
    
    # validate columns
    required_columns = ['reservation_status_date', 'stays_in_week_nights', 'stays_in_weekend_nights', 'adr']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")
    
    # missing values filled
    df.fillna({
        'children': 0,
        'country': 'unknown',
        'agent': 0,
        'company': 0
    }, inplace=True)
    
    # convert date columns
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')

    
    # drop rows with missing reservation dates
    initial_rows = len(df)
    df.dropna(subset=['reservation_status_date'], inplace=True)
    cleaned_rows = len(df)
    print(f"dropped {initial_rows - cleaned_rows} rows with missing reservation dates.")
    
    # save to cleaned_hotel_bookings.csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"data cleaned and saved to {output_path}")


if __name__ == "__main__":
    input_path = "data/hotel_bookings.csv"
    output_path = "data/cleaned_hotel_bookings.csv"
    preprocess_data(input_path, output_path)