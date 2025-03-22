# this script analyzes hotel booking data with visualizations for revenue, cancellations, geography, and lead times
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# load data
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")
    return pd.read_csv(path, parse_dates=['reservation_status_date'])


# plots monthly revenue trend
def revenue_trends(df):
    df['total_revenue'] = (df['stays_in_week_nights'] + df['stays_in_weekend_nights']) * df['adr']
    monthly_revenue = df.groupby(df['reservation_status_date'].dt.to_period('M'))['total_revenue'].sum()
    plt.figure(figsize=(8, 4))
    monthly_revenue.plot(kind='line', title='monthly revenue trend', ylabel='total revenue', xlabel='month', linewidth=2, color='royalblue')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# calculates and prints cancellation rate
def cancellation_rate(df):
    total_bookings = len(df)
    cancellations = df['is_canceled'].sum()
    rate = (cancellations / total_bookings) * 100
    print(f"cancellation rate: {rate:.2f}%")


# plots top 10 countries by bookings
def geographical_distribution(df):
    country_counts = df['country'].value_counts().head(10)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=country_counts.values, y=country_counts.index, hue=country_counts.index, legend=False)
    plt.title('top 10 countries by number of bookings')
    plt.xlabel('number of bookings')
    plt.ylabel('country')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# plots lead time distribution
def lead_time_distribution(df):
    plt.figure(figsize=(8, 4))
    sns.histplot(df['lead_time'], bins=50, kde=True, color='darkorange')
    plt.title('booking lead time distribution')
    plt.xlabel('lead time (days)')
    plt.ylabel('number of bookings')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    data_path = "data/cleaned_hotel_bookings.csv"
    df = load_data(data_path)

    print("generating revenue trends...")
    revenue_trends(df)
    
    print("calculating cancellation rate...")
    cancellation_rate(df)
    
    print("plotting geographical distribution...")
    geographical_distribution(df)
    
    print("plotting lead time distribution...")
    lead_time_distribution(df)
    print("analytics generation complete.")