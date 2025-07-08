import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 确保正确的应用初始化
def main():
    # 设置页面配置（必须在其他streamlit调用之前）
    st.set_page_config(
        page_title="模型选择应用",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 检查session_state是否正确初始化
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.user_data = {}

    # 主应用逻辑
    st.title("模型选择应用")
    st.write("应用正在运行...")

    # 你的其他代码...


# 添加错误处理
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"应用启动错误: {str(e)}")
        st.info("请刷新页面重试")
# 设置Matplotlib字体以支持中文显示
# 对于本地测试，请确保系统安装了SimHei字体。
# 在某些云环境中，可能需要额外配置字体，否则会回退到默认字体。
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    st.warning("无法为Matplotlib设置中文字体。图表显示可能无法正确显示中文字符。")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 回退方案
    plt.rcParams['axes.unicode_minus'] = False

# --- Streamlit页面配置 ---
st.set_page_config(layout="wide", page_title="病例数预测应用")

st.title("病例数时间序列预测")
st.markdown("使用深度学习模型（LSTM, GRU, CNN-LSTM）预测病例数。")


## 数据加载与初步处理

@st.cache_data
def load_data():
    df = pd.read_csv('heal2.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    return df

df = load_data()

st.subheader("数据概览")
st.write(df.head())
st.write(f"数据形状: {df.shape}")
st.write(f"时间范围: {df['ds'].min().strftime('%Y-%m-%d')} 到 {df['ds'].max().strftime('%Y-%m-%d')}")
st.write(f"病例数范围: {df['y'].min()} 到 {df['y'].max()}")

# 原始数据可视化（可选在Streamlit中显示，这里仅用于展示数据特性）
st.subheader("原始数据可视化")
fig_data, axes_data = plt.subplots(2, 2, figsize=(15, 10))

axes_data[0, 0].plot(df['ds'], df['y'], linewidth=2, color='red')
axes_data[0, 0].set_title('病例数时间序列', fontsize=14, fontweight='bold')
axes_data[0, 0].set_xlabel('日期')
axes_data[0, 0].set_ylabel('病例数')
axes_data[0, 0].grid(True, alpha=0.3)

axes_data[0, 1].scatter(df['temperature'], df['y'], alpha=0.6, color='orange')
axes_data[0, 1].set_title('温度 vs 病例数', fontsize=14, fontweight='bold')
axes_data[0, 1].set_xlabel('温度 (°C)')
axes_data[0, 1].set_ylabel('病例数')
axes_data[0, 1].grid(True, alpha=0.3)

axes_data[1, 0].scatter(df['humidity'], df['y'], alpha=0.6, color='blue')
axes_data[1, 0].set_title('湿度 vs 病例数', fontsize=14, fontweight='bold')
axes_data[1, 0].set_xlabel('湿度')
axes_data[1, 0].set_ylabel('病例数')
axes_data[1, 0].grid(True, alpha=0.3)

weekend_cases = df.groupby('is_weekend')['y'].mean()
axes_data[1, 1].bar(['工作日', '周末'], weekend_cases.values, color=['lightblue', 'lightcoral'])
axes_data[1, 1].set_title('工作日 vs 周末平均病例数', fontsize=14, fontweight='bold')
axes_data[1, 1].set_ylabel('平均病例数')
axes_data[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig_data)




## 特征工程函数


def create_features(df_input):
    df_copy = df_input.copy()

    # 时间特征
    df_copy['day_of_year'] = df_copy['ds'].dt.dayofyear
    df_copy['month'] = df_copy['ds'].dt.month
    df_copy['day_of_week'] = df_copy['ds'].dt.dayofweek
    df_copy['week_of_year'] = df_copy['ds'].dt.isocalendar().week.astype(int) # 确保为整数类型

    # 周期性特征
    df_copy['sin_day'] = np.sin(2 * np.pi * df_copy['day_of_year'] / 365.25)
    df_copy['cos_day'] = np.cos(2 * np.pi * df_copy['day_of_year'] / 365.25)
    df_copy['sin_week'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
    df_copy['cos_week'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)

    # 滞后特征
    for lag in [1, 3, 7, 14]:
        df_copy[f'y_lag_{lag}'] = df_copy['y'].shift(lag)

    # 移动平均特征
    for window in [3, 7, 14]:
        df_copy[f'y_ma_{window}'] = df_copy['y'].rolling(window=window).mean()
        df_copy[f'temp_ma_{window}'] = df_copy['temperature'].rolling(window=window).mean()

    # 填充NaN值 (bfill后ffill确保双向填充)
    df_copy = df_copy.fillna(method='bfill').fillna(method='ffill')

    return df_copy

df_features = create_features(df.copy()) # 使用 df 的副本进行特征工程

# 定义特征列
feature_cols = ['temperature', 'humidity', 'is_weekend', 'school_holiday',
                'day_of_year', 'month', 'day_of_week', 'week_of_year',
                'sin_day', 'cos_day', 'sin_week', 'cos_week',
                'y_lag_1', 'y_lag_3', 'y_lag_7', 'y_lag_14',
                'y_ma_3', 'y_ma_7', 'y_ma_14', 'temp_ma_3', 'temp_ma_7', 'temp_ma_14']


def create_sequences(data, target, feature_cols, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[feature_cols].iloc[i-seq_length:i].values)
        y.append(target.iloc[i])
    return np.array(X), np.array(y)
def build_lstm_model(input_shape, learning_rate, dropout):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(64, return_sequences=True),
        Dropout(dropout),
        LSTM(32, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='huber', metrics=['mae'])
    return model

def build_gru_model(input_shape, learning_rate, dropout):
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        GRU(64, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='huber', metrics=['mae'])
    return model

def build_cnn_lstm_model(input_shape, learning_rate, dropout):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=True),
        Dropout(dropout),
        LSTM(25, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='huber', metrics=['mae'])
    return model
st.sidebar.header("模型配置")

selected_model_name = st.sidebar.selectbox(
    "选择模型:",
    ('LSTM', 'GRU', 'CNN-LSTM')
)

st.sidebar.subheader("模型超参数")
seq_length = st.sidebar.slider("序列长度 (天):", 7, 30, 14)
epochs = st.sidebar.slider("训练轮次 (Epochs):", 10, 200, 100)
batch_size = st.sidebar.slider("批次大小 (Batch Size):", 16, 128, 32)
learning_rate = st.sidebar.slider("学习率 (Learning Rate):", 0.0001, 0.01, 0.001, format="%.4f")
dropout_rate = st.sidebar.slider("Dropout 比率:", 0.0, 0.5, 0.2, format="%.2f")

st.sidebar.subheader("回调函数参数")
es_patience = st.sidebar.slider("早停耐心 (Early Stopping Patience):", 5, 50, 20)
rlr_patience = st.sidebar.slider("学习率下降耐心 (RLRPatience):", 5, 30, 10)
rlr_factor = st.sidebar.slider("学习率下降因子 (RLRFactor):", 0.1, 0.9, 0.5, format="%.1f")

train_button = st.sidebar.button("训练并评估模型")
if train_button:
    st.subheader(f"正在训练和评估 {selected_model_name} 模型...")
    st.markdown("请耐心等待，训练可能需要一些时间。")

    with st.spinner('数据准备中...'):
        # 数据标准化 (由于seq_length可能改变数据形状，需要重新fit)
        scaler_X = StandardScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(df_features[feature_cols])
        y_scaled = scaler_y.fit_transform(df_features[['y']])

        df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        df_scaled['y'] = y_scaled

        # 根据选择的序列长度创建序列数据
        X_seq, y_seq = create_sequences(df_scaled, df_scaled['y'], feature_cols, seq_length)

        # 训练集和测试集分割
        train_size = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]

        st.success("数据准备完成！")
        st.write(f"训练集形状: X_train: {X_train.shape}, y_train: {y_train.shape}")
        st.write(f"测试集形状: X_test: {X_test.shape}, y_test: {y_test.shape}")

    with st.spinner(f'正在训练 {selected_model_name} 模型...'):
        # 构建选定的模型
        input_shape = (X_train.shape[1], X_train.shape[2])
        if selected_model_name == 'LSTM':
            model = build_lstm_model(input_shape, learning_rate, dropout_rate)
        elif selected_model_name == 'GRU':
            model = build_gru_model(input_shape, learning_rate, dropout_rate)
        else: # CNN-LSTM
            model = build_cnn_lstm_model(input_shape, learning_rate, dropout_rate)

        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=es_patience, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=rlr_factor, patience=rlr_patience, min_lr=0.0001, verbose=0)
        ]

        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0 # 抑制控制台的详细输出，Streamlit会处理进度
        )
        st.success(f"{selected_model_name} 模型训练完成！")

    with st.spinner('正在评估模型并进行预测...'):
        # 预测
        train_pred = model.predict(X_train, verbose=0)
        test_pred = model.predict(X_test, verbose=0)

        # 反标准化，并转换为整数
        train_pred_orig = np.maximum(0, np.round(scaler_y.inverse_transform(train_pred.reshape(-1, 1)))).astype(int).flatten()
        test_pred_orig = np.maximum(0, np.round(scaler_y.inverse_transform(test_pred.reshape(-1, 1)))).astype(int).flatten()
        y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # 计算评估指标
        train_mae = mean_absolute_error(y_train_orig, train_pred_orig)
        test_mae = mean_absolute_error(y_test_orig, test_pred_orig)
        train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred_orig))
        test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))
        train_r2 = r2_score(y_train_orig, train_pred_orig)
        test_r2 = r2_score(y_test_orig, test_pred_orig)

        st.subheader(f"{selected_model_name} 模型评估结果")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("训练集 MAE", f"{train_mae:.2f}")
            st.metric("测试集 MAE", f"{test_mae:.2f}")
        with col2:
            st.metric("训练集 RMSE", f"{train_rmse:.2f}")
            st.metric("测试集 RMSE", f"{test_rmse:.2f}")
        with col3:
            st.metric("训练集 R²", f"{train_r2:.3f}")
            st.metric("测试集 R²", f"{test_r2:.3f}")

        st.subheader("训练损失历史")
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        ax_loss.plot(history.history['loss'], label='训练集 Loss')
        ax_loss.plot(history.history['val_loss'], label='验证集 Loss', linestyle='--')
        ax_loss.set_title(f'{selected_model_name} 模型训练损失', fontsize=14, fontweight='bold')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        st.pyplot(fig_loss)

        st.subheader(f"{selected_model_name} 模型预测结果")
        fig_pred, ax_pred = plt.subplots(figsize=(15, 8))

        # 准备绘图数据
        train_dates = df['ds'].iloc[seq_length:seq_length+len(train_pred_orig)]
        test_dates = df['ds'].iloc[seq_length+len(train_pred_orig):]
        train_actual = df['y'].iloc[seq_length:seq_length+len(train_pred_orig)]
        test_actual = df['y'].iloc[seq_length+len(train_pred_orig):]

        ax_pred.plot(train_dates, train_actual, 'b-', label='训练集实际', linewidth=2)
        ax_pred.plot(test_dates, test_actual, 'g-', label='测试集实际', linewidth=2)
        ax_pred.plot(train_dates, train_pred_orig, 'r--', label='训练集预测', alpha=0.8)
        ax_pred.plot(test_dates, test_pred_orig, 'orange', linestyle='--', label='测试集预测', alpha=0.8)

        ax_pred.set_title(f'{selected_model_name} 模型实际与预测', fontsize=14, fontweight='bold')
        ax_pred.set_xlabel('日期')
        ax_pred.set_ylabel('病例数')
        ax_pred.legend(fontsize=10)
        ax_pred.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig_pred)

        st.subheader("未来28天预测")

        # --- 未来28天预测逻辑 ---
        # 准备用于预测的最后一个序列
        last_sequence_data = df_scaled[feature_cols].iloc[-seq_length:].values
        last_sequence = last_sequence_data.reshape(1, seq_length, len(feature_cols))

        future_predictions = []
        last_date = df['ds'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=28, freq='D')

        # 创建一个DataFrame来存储未来特征，方便按列名访问
        future_features_df = pd.DataFrame(index=future_dates, columns=feature_cols)

        # 填充静态的未来特征（温度/湿度估算、时间特征、周末/节假日）
        for date in future_dates:
            day_of_year = date.dayofyear
            # 简化的未来温度/湿度估算（可根据实际情况替换为更精确的预测）
            temp_estimate = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
            humidity_estimate = 0.65
            is_weekend = 1 if date.weekday() >= 5 else 0
            school_holiday = 0 # 假设未来非假期

            future_features_df.loc[date, ['temperature', 'humidity', 'is_weekend', 'school_holiday']] = \
                [temp_estimate, humidity_estimate, is_weekend, school_holiday]

            future_features_df.loc[date, 'day_of_year'] = day_of_year
            future_features_df.loc[date, 'month'] = date.month
            future_features_df.loc[date, 'day_of_week'] = date.weekday()
            future_features_df.loc[date, 'week_of_year'] = date.isocalendar().week
            future_features_df.loc[date, 'sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
            future_features_df.loc[date, 'cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)
            future_features_df.loc[date, 'sin_week'] = np.sin(2 * np.pi * date.weekday() / 7)
            future_features_df.loc[date, 'cos_week'] = np.cos(2 * np.pi * date.weekday() / 7)

        # 迭代预测并更新滞后/移动平均特征
        temp_historical_y = df['y'].tolist() # 保留历史y值
        temp_future_y_predictions = [] # 存储预测的y值，用于计算未来的滞后和移动平均

        for i in range(28):
            current_date_features = future_features_df.iloc[i].copy()

            # 更新滞后特征：优先使用已预测的未来值，不足则使用历史值
            current_date_features['y_lag_1'] = temp_future_y_predictions[-1] if i >= 1 else temp_historical_y[-1]
            current_date_features['y_lag_3'] = temp_future_y_predictions[-3] if i >= 3 else temp_historical_y[max(0, len(temp_historical_y)-3)]
            current_date_features['y_lag_7'] = temp_future_y_predictions[-7] if i >= 7 else temp_historical_y[max(0, len(temp_historical_y)-7)]
            current_date_features['y_lag_14'] = temp_future_y_predictions[-14] if i >= 14 else temp_historical_y[max(0, len(temp_historical_y)-14)]

            # 更新移动平均特征：结合历史和已预测的未来值计算
            combined_y_for_ma = temp_historical_y + temp_future_y_predictions
            current_date_features['y_ma_3'] = np.mean(combined_y_for_ma[-3:]) if len(combined_y_for_ma) >= 3 else combined_y_for_ma[-1]
            current_date_features['y_ma_7'] = np.mean(combined_y_for_ma[-7:]) if len(combined_y_for_ma) >= 7 else combined_y_for_ma[-1]
            current_date_features['y_ma_14'] = np.mean(combined_y_for_ma[-14:]) if len(combined_y_for_ma) >= 14 else combined_y_for_ma[-1]

            # 简化的温度移动平均（若有温度预测可更复杂）
            current_date_features['temp_ma_3'] = current_date_features['temperature']
            current_date_features['temp_ma_7'] = current_date_features['temperature']
            current_date_features['temp_ma_14'] = current_date_features['temperature']

            # 对当前特征进行标准化
            new_features_array = np.array([current_date_features[col] for col in feature_cols]).reshape(1, -1)
            new_features_scaled = scaler_X.transform(new_features_array)

            # 更新 last_sequence 用于下一次预测：左移一个时间步，并加入新的特征
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, :] = new_features_scaled[0]

            # 预测下一个值
            pred_scaled = model.predict(last_sequence, verbose=0)[0, 0]
            pred_orig = scaler_y.inverse_transform([[pred_scaled]])[0, 0]

            # 转换为整数并确保非负
            rounded_pred = np.maximum(0, np.round(pred_orig)).astype(int)

            future_predictions.append(rounded_pred)
            temp_future_y_predictions.append(rounded_pred) # 将四舍五入后的整数值加入列表

        future_df = pd.DataFrame({
            '日期': future_dates,
            '预测病例数': future_predictions # 此列表现在包含整数
        })

        st.dataframe(future_df)

        # 可视化未来预测
        fig_future, ax_future = plt.subplots(figsize=(15, 8))
        ax_future.plot(df['ds'], df['y'], 'b-', label='历史病例数', linewidth=2)
        ax_future.plot(future_dates, future_predictions, 'r-', label='未来28天预测', linewidth=2, marker='o')

        # 添加预测区间（使用简化的置信区间，基于近期历史的标准差）
        pred_std_hist = np.std(df['y'].tail(seq_length)) # 使用近期实际值的标准差
        lower_bound = np.array(future_predictions) - 1.96 * pred_std_hist
        upper_bound = np.array(future_predictions) + 1.96 * pred_std_hist
        # 确保置信区间下限不为负
        ax_future.fill_between(future_dates, np.maximum(0, lower_bound), upper_bound,
                                alpha=0.3, color='red', label='95% 置信区间')

        ax_future.axvline(x=df['ds'].iloc[-1], color='green', linestyle='--', alpha=0.7, label='预测起始点')
        ax_future.set_title('病例数预测 - 未来28天', fontsize=16, fontweight='bold')
        ax_future.set_xlabel('日期')
        ax_future.set_ylabel('病例数')
        ax_future.legend()
        ax_future.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig_future)

        st.subheader("预测趋势分析")
        avg_prediction = np.mean(future_predictions)
        if avg_prediction > df['y'].tail(7).mean():
            st.markdown("⚠️ **预测显示未来病例数呈上升趋势**")
        else:
            st.markdown("✅ **预测显示未来病例数相对稳定或下降**")

        st.write(f"未来28天平均预测病例数: {avg_prediction:.1f}")
        st.write(f"预测峰值: {max(future_predictions):.0f} (在未来第 {future_predictions.index(max(future_predictions))+1} 天)")
        st.write(f"预测最低值: {min(future_predictions):.0f} (在未来第 {future_predictions.index(min(future_predictions))+1} 天)")