import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # 1. Swap checkIn/Out if inverted
    invalid_dates = df[df['checkOutDate'] < df['checkInDate']]
    df.loc[invalid_dates.index, ['checkInDate', 'checkOutDate']] = df.loc[invalid_dates.index, ['checkOutDate', 'checkInDate']].values

    # 2. Remove stays > 90 days
    df = df[df['checkOutDate'] - df['checkInDate'] <= pd.Timedelta(days=90)]

    # 3. Ensure searchDate < checkInDate
    df.loc[df['searchDate'] > df['checkInDate'], 'searchDate'] = df['checkInDate'] - pd.Timedelta(days=1)

    # 4. Handle missing destinationName
    df['destinationName'].fillna('UNKNOWN', inplace=True)

    # 5. Replace -1 with NaN for numerical columns
    cols_to_fix = ['starLevel', 'customerReviewScore', 'reviewCount']
    df[cols_to_fix] = df[cols_to_fix].replace(-1, np.nan)

    # 6. Cap prices at 99th percentile
    df['minPrice'] = df['minPrice'].clip(upper=df['minPrice'].quantile(0.99))
    df['minStrikePrice'] = df['minStrikePrice'].clip(upper=df['minStrikePrice'].quantile(0.99))

    # 7. Ensure rank is always ≥ 1
    df['rank'] = df['rank'].clip(lower=1)

    # 8. Feature Engineering
    df['stayLength'] = (df['checkOutDate'] - df['checkInDate']).dt.days
    df['leadTime'] = (df['checkInDate'] - df['searchDate']).dt.days
    df['weekendStay'] = ((df['checkInDate'].dt.weekday >= 4) | (df['checkOutDate'].dt.weekday >= 4)).astype(int)
    df['highRank'] = (df['rank'] <= 10).astype(int)
    df['discountPercent'] = ((df['minStrikePrice'] - df['minPrice']) / df['minStrikePrice']).fillna(0).clip(0, 1)
    df['logMinPrice'] = np.log1p(df['minPrice'])
    df['checkInDayOfWeek'] = df['checkInDate'].dt.dayofweek

    # 9. Encode categorical features
    df = pd.get_dummies(df, columns=['vipTier', 'deviceCode'], drop_first=True)
    dest_freq = df['destinationName'].value_counts(normalize=True)
    df['destinationFreq'] = df['destinationName'].map(dest_freq)

    # 10. Drop columns NOT usable for ML
    df.drop(columns=[
        'checkInDate', 'checkOutDate', 'searchDate',
        'destinationName', 'userId', 'hotelId'
    ], inplace=True)

    # Drop clickLabel from model input (fix data leakage)
    if 'clickLabel' in df.columns:
        df.drop(columns=['clickLabel'], inplace=True)

    # 11. Normalize numerical columns
    scaler = StandardScaler()
    num_cols = ['stayLength', 'leadTime', 'minPrice', 'minStrikePrice', 'customerReviewScore', 'reviewCount']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 12. Drop rows with missing values
    df.dropna(inplace=True)

    # 13. Ranking Label Creation (0 = shown only, 1 = booked)
    df['ranking_label'] = 0
    df.loc[df['bookingLabel'] == 1, 'ranking_label'] = 1
    
    return df
