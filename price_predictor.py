import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
from datetime import datetime, timedelta

class PricePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.target_scaler = StandardScaler()
        self.is_trained = False
        self.weather_adjustments = {
            'Rainy': -0.05,
            'Humidity': -0.03,
            'Sunny': 0.02,
            'Cloudy': 0.00,
            'Snowy': -0.08
        }
        
    def _calculate_demand_level(self, df):
        """Calculate demand level if not present"""
        if 'demand_level' not in df.columns and 'raw_demand_score' in df.columns:
            df['demand_level'] = pd.cut(
                df['raw_demand_score'],
                bins=[0, 0.4, 0.7, 1],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
        return df

    def _apply_weather_adjustment(self, df):
        """Apply price adjustments based on weather conditions"""
        if 'weather' in df.columns:
            df['weather_adjustment'] = df['weather'].map(self.weather_adjustments).fillna(0)
        else:
            df['weather_adjustment'] = 0
        return df

    def _apply_shelf_life_discount(self, df):
        """Apply discounts based on shelf life and days to expiry"""
        if 'days_to_expiry' in df.columns:
            # Calculate discount based on days to expiry
            df['shelf_life_discount'] = np.select(
                [
                    df['days_to_expiry'] <= 7,
                    df['days_to_expiry'] <= 15,
                    df['days_to_expiry'] <= 30
                ],
                [-0.25, -0.15, -0.10],
                default=0
            )
            
            # Automatically activate promotion for items expiring soon
            if 'promotion_active' not in df.columns:
                df['promotion_active'] = False
            df['promotion_active'] = np.where(
                df['days_to_expiry'] <= 30,
                True,
                df['promotion_active']
            )
        else:
            df['shelf_life_discount'] = 0
            
        return df

    def _calculate_price_adjustments(self, df):
        """Calculate manual price adjustments based on business rules"""
        df = self._calculate_demand_level(df)
        df = self._apply_weather_adjustment(df)
        df = self._apply_shelf_life_discount(df)
        
        # Initialize all adjustments to 0
        for adj in ['demand', 'stock', 'expiry', 'promotion', 'competitor']:
            df[f'{adj}_adjustment'] = 0.0

        # Demand adjustment
        if 'demand_level' in df.columns:
            df['demand_adjustment'] = np.select(
                [
                    df['demand_level'] == 'High',
                    df['demand_level'] == 'Low'
                ],
                [0.05, -0.05],
                default=0
            )

        # Stock adjustment
        if all(col in df.columns for col in ['stock_units', 'past_sales']):
            df['stock_adjustment'] = np.where(
                (df['stock_units'] > df['past_sales'] * 3) & (df['past_sales'] < df['stock_units'] * 0.3),
                -0.10,
                np.where(
                    (df['stock_units'] < df['past_sales']) & (df['past_sales'] > df['stock_units'] * 1.5),
                    0.10,
                    0
                )
            )

        # Expiry adjustment (now combined with shelf life discount)
        df['expiry_adjustment'] = df['shelf_life_discount']

        # Promotion adjustment
        if 'promotion_active' in df.columns:
            df['promotion_adjustment'] = np.where(
                df['promotion_active'],
                -0.10,
                0
            )

        # Competitor adjustment
        if all(col in df.columns for col in ['base_price', 'competitor_price']):
            df['competitor_adjustment'] = np.where(
                df['base_price'] > df['competitor_price'],
                -0.03,
                np.where(
                    df['base_price'] < df['competitor_price'],
                    0.02,
                    0
                )
            )

        # Calculate total adjustment (including weather)
        df['total_adjustment'] = df[[
            'demand_adjustment', 'stock_adjustment', 'expiry_adjustment',
            'promotion_adjustment', 'competitor_adjustment', 'weather_adjustment'
        ]].sum(axis=1)

        # Apply margin protection
        if 'cost_price' in df.columns:
            min_margin = 0.15  # 15% minimum margin
            df['min_price'] = df['cost_price'] / (1 - min_margin)
            df['suggested_price'] = np.maximum(
                df['base_price'] * (1 + df['total_adjustment']),
                df['min_price']
            )
        else:
            df['suggested_price'] = df['base_price'] * (1 + df['total_adjustment'])
        
        return df

    def train_model(self, df):
        try:
            # Calculate manual adjustments
            df = self._calculate_price_adjustments(df)
            
            # Prepare features and target
            X = df.drop(['product_id', 'suggested_price'], axis=1, errors='ignore')
            y = df['suggested_price']
            
            # Scale target variable
            y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
            
            # Identify feature types
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Ensure weather is included in categorical features
            if 'weather' not in categorical_cols and 'weather' in X.columns:
                categorical_cols.append('weather')
            
            # Create preprocessing pipeline
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ])
            
            # Create and train model
            self.model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    random_state=42
                ))
            ])
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_scaled, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            preds = self.model.predict(X_test)
            preds = self.target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            y_test_orig = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            mae = mean_absolute_error(y_test_orig, preds)
            print(f"Model trained with MAE: {mae:.2f}")
            
            self.is_trained = True
            return mae
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise e
    
    def predict_prices(self, df):
        if not self.is_trained:
            # If model not trained, use rule-based pricing
            df = self._calculate_price_adjustments(df)
            return df['suggested_price']
            
        try:
            # Calculate manual adjustments for reference
            df = self._calculate_price_adjustments(df.copy())
            
            # Prepare features
            X = df.drop(['product_id', 'suggested_price'], axis=1, errors='ignore')
            
            # Make predictions
            predicted_scaled = self.model.predict(X)
            predicted_prices = self.target_scaler.inverse_transform(
                predicted_scaled.reshape(-1, 1)
            ).flatten()
            
            # Apply margin protection to predictions
            if 'cost_price' in df.columns:
                min_margin = 0.15
                min_prices = df['cost_price'] / (1 - min_margin)
                predicted_prices = np.maximum(predicted_prices, min_prices)
            
            return predicted_prices
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise e
    
    def save_model(self, path='price_model.pkl'):
        joblib.dump({
            'model': self.model,
            'target_scaler': self.target_scaler,
            'preprocessor': self.preprocessor,
            'is_trained': self.is_trained
        }, path)
    
    def load_model(self, path='price_model.pkl'):
        saved = joblib.load(path)
        self.model = saved['model']
        self.target_scaler = saved['target_scaler']
        self.preprocessor = saved['preprocessor']
        self.is_trained = saved['is_trained']
        return self