import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

class DataProcessor:
    def __init__(self):
        self.preprocessor = None
        self.target_scaler = StandardScaler()
        self.categorical_cols = ['category', 'product_type', 'product_label', 'weather', 
                               'country', 'location', 'season', 'inventory_type',
                               'stock_status', 'expiry_status', 'demand_level']
        self.numerical_cols = ['raw_demand_score', 'demand', 'base_price', 'stock_units',
                             'past_sales', 'shelf_life', 'competitor_price', 'cost_price',
                             'days_to_expiry', 'price_adjustment']
        
    def _calculate_days_to_expiry(self, row):
        """Calculate days until product expiry"""
        if pd.isna(row['manufacture_date']) or pd.isna(row['shelf_life']):
            return np.nan
        expiry_date = row['manufacture_date'] + pd.Timedelta(days=row['shelf_life'])
        return (expiry_date - datetime.now()).days
    
    def _validate_weather_data(self, df):
        """Ensure weather data is valid"""
        valid_weather_types = ['Rainy', 'Humidity', 'Sunny', 'Cloudy', 'Snowy']
        if 'weather' in df.columns:
            # Replace invalid weather types with most common valid type
            invalid_mask = ~df['weather'].isin(valid_weather_types)
            if invalid_mask.any():
                most_common = df.loc[df['weather'].isin(valid_weather_types), 'weather'].mode()[0]
                df.loc[invalid_mask, 'weather'] = most_common
        else:
            # Add weather column with default value if missing
            df['weather'] = 'Sunny'
        return df
    
    def _apply_business_logic(self, df):
        """Apply all business rules to the dataframe"""
        df = self._validate_weather_data(df)
        
        # Stock status
        df['stock_status'] = np.where(df['stock_units'] < 30, '⚠ Low Stock', 'Adequate Stock')
        
        # Calculate days to expiry
        df['days_to_expiry'] = df.apply(self._calculate_days_to_expiry, axis=1)
        df['expiry_status'] = np.where(
            df['days_to_expiry'] < 30, 
            '⏳ Soon Expiring', 
            'Normal'
        )
        
        # Automatic promotion activation for items expiring soon
        if 'promotion_active' not in df.columns:
            df['promotion_active'] = False
        df['promotion_active'] = np.where(
            df['days_to_expiry'] <= 30,
            True,
            df['promotion_active']
        )
        
        # Demand level calculation
        df['demand_level'] = pd.cut(
            df['raw_demand_score'],
            bins=[0, 0.4, 0.7, 1],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        return df
    
    def clean_data(self, df):
        """Handle missing values and apply business logic"""
        # Convert dates
        if 'manufacture_date' in df.columns:
            df['manufacture_date'] = pd.to_datetime(df['manufacture_date'])
        
        # Apply business rules
        df = self._apply_business_logic(df)
        
        # Fill missing numerical values with median
        for col in self.numerical_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill missing categorical values with mode
        for col in self.categorical_cols:
            if col in df.columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def create_preprocessor(self, df):
        """Create the preprocessing pipeline"""
        existing_cat_cols = [col for col in self.categorical_cols if col in df.columns]
        existing_num_cols = [col for col in self.numerical_cols if col in df.columns]
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, existing_num_cols),
                ('cat', categorical_transformer, existing_cat_cols)
            ])
        
        return self.preprocessor
    
    def prepare_features(self, df):
        """Prepare features for model training/prediction"""
        df_clean = self.clean_data(df.copy())
        
        # Ensure all expected columns exist
        for col in self.numerical_cols + self.categorical_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0 if col in self.numerical_cols else 'Unknown'
                
        return df_clean
    
    def scale_target(self, y, fit=False):
        """Scale or inverse scale the target variable"""
        if fit:
            return self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        else:
            return self.target_scaler.inverse_transform(y.values.reshape(-1, 1)).flatten()