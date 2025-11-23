import numpy as np
import pandas as pd
import tensorflow as tf
import polars as pl
import itertools

from Deep_Learning_LSTM_Regression_Window_Generator import WindowGenerator
from Deep_Learning_LSTM_Regression_Baseline import Baseline
from Deep_Learning_LSTM_Regression_Plots import plot_model_vs_baseline_vs_actual

filepath = "./9417_imputed_data"

# extract base datafiles without temporal data
cols = [
   'DateTime', 'Date', 'Time', 'linear_dt', 'sin_time_of_year',
   'cos_time_of_year', 'is_mon', 'is_tue', 'is_wed', 'is_thu', 'is_fri',
   'is_sat', 'is_sun', 'is_weekday', 'sin_time_of_week',
   'cos_time_of_week', 'sin_hour_of_day', 'cos_hour_of_day', 'CO(GT)',
   'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)',
   'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH','AH'
]
df_pl = pl.read_ipc(filepath + "/Z-score normalised, with lags.parquet").select(cols).to_pandas()
df = df_pl.dropna()

# Drop date and time (Already combined in single timestamp)
df = df.drop(columns=["Date", "Time"], errors="ignore")

# GT pollutants to predict
TARGET_COLS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]

# All input features except GT and DateTime (since we will drop DateTime later)
INPUT_FEATURES = [
    c for c in df.columns
    if c not in ["DateTime"] + TARGET_COLS   
]

TRAINING_ROWS = 6043
VALIDATION_ROWS = 1067

HORIZONS = [1, 6, 12, 24]
TARGET_WIDTH = 1
BATCH_SIZE = 32
EPOCHS = 20

# Hyperparameters pool
WINDOW_SIZE = [24, 48, 72, 168]
LSTM_UNITS = [64, 128]
DENSE_UNITS = [32, 64]
DROPOUTS = [0.0, 0.2]
LR = [5e-4, 1e-3]


def train_val_test_split(df):
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime")

    # Set 2005 as test_data
    test_mask = df["DateTime"].dt.year == 2005

    train_val_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    # 85% Training and 15% Validation
    train_df = train_val_df.iloc[:TRAINING_ROWS].copy()
    val_df = train_val_df.iloc[TRAINING_ROWS:TRAINING_ROWS + VALIDATION_ROWS].copy()

    return train_df, val_df, test_df


def make_rnn(input_width, num_features, lstm_units, dropout,
             dense_units, num_outputs, lr, activation="relu"):
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_width, num_features)),
        tf.keras.layers.LSTM(lstm_units, dropout=dropout, return_sequences=False),
        tf.keras.layers.Dense(dense_units, activation=activation),
        tf.keras.layers.Dense(num_outputs)                      # All pollutants are outputs
    ])

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),                # Take SQRT to find RMSE
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),   # Could also use RMSProp / SGD
    )

    return model   


def main():

    # Split the data to 85% Training and 15% Validation yr=2005 Test
    train_df, val_df, test_df = train_val_test_split(df)

    train_inputs = train_df[INPUT_FEATURES]
    val_inputs = val_df[INPUT_FEATURES]
    test_inputs = test_df[INPUT_FEATURES]

    train_targets = train_df[TARGET_COLS]
    val_targets = val_df[TARGET_COLS]
    test_targets = test_df[TARGET_COLS]

    final_results = {}


    # Hyperparameters grid search per horizon
    for horizon in HORIZONS:

        print("\n" + "=" * 90)
        print(f"\t\t\t\t\tT = {horizon} hours")
        print("=" * 90)

        best_horizon_hyperparams = None
        best_horizon_hyperparams_val_loss = np.inf     

        for input_width in WINDOW_SIZE:

            print("\n" + "-" * 90)
            print(f"\t\t\t\tSearching WINDOW SIZE = {input_width} hours")
            print("-" * 90)

            # Building a window for this iteration of searching
            window = WindowGenerator(
                input_width=input_width,
                target_width=TARGET_WIDTH,
                shift=horizon,
                input_train_df=train_inputs,
                input_val_df=val_inputs,
                input_test_df=test_inputs,
                target_train_df=train_targets,
                target_val_df=val_targets,
                target_test_df=test_targets,
                target_cols=TARGET_COLS,
                input_feature_cols=INPUT_FEATURES,
                batch_size=BATCH_SIZE
            )

            # Search hyperparameters for this window hyperparameter
            for lstm_units, dense_units, dropout, lr in itertools.product(
                LSTM_UNITS, DENSE_UNITS, DROPOUTS, LR
            ):
                model = make_rnn(
                    input_width=input_width,
                    num_features=len(INPUT_FEATURES),
                    lstm_units=lstm_units,
                    dropout=dropout,
                    dense_units=dense_units,
                    num_outputs=len(TARGET_COLS),
                    lr=lr
                )

                # To stop model when there is no improvements (minimizes overfitting
                # reliefs us from setting EPOCHS as a hyperparameter)
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True
                )

                # Train the model with the training set and validate against the 
                # validation set
                model.fit(
                    window.train,
                    epochs=EPOCHS,
                    validation_data=window.val,
                    callbacks=[early_stop],
                    # Silence output since there are many hyperparameters
                    verbose=0
                )

                curr_hyperparams_val_results = model.evaluate(window.val, verbose=0, return_dict=True)
                curr_hyperparams_val_loss = curr_hyperparams_val_results["loss"]

                print("~" * 90)
                print(f"\t\tSearching paramters: LSTM={lstm_units}, Dense={dense_units}, Dropout={dropout}, LR={lr}")  
                print(f"\t\t- Validation loss: {curr_hyperparams_val_loss:.4f}")
                print("~" * 90)

                if curr_hyperparams_val_loss < best_horizon_hyperparams_val_loss:
                    best_horizon_hyperparams_val_loss = curr_hyperparams_val_loss
                    best_horizon_hyperparams = {
                        "input_width": input_width,
                        "lstm_units": lstm_units,
                        "dense_units": dense_units,
                        "dropout": dropout,
                        "lr": lr
                    }

        print("\n\nBest config for this horizon:", best_horizon_hyperparams)

        best_window = WindowGenerator(
            input_width=best_horizon_hyperparams["input_width"],
            target_width=TARGET_WIDTH,
            shift=horizon,
            input_train_df=train_inputs,
            input_val_df=val_inputs,
            input_test_df=test_inputs,
            target_train_df=train_targets,
            target_val_df=val_targets,
            target_test_df=test_targets,
            target_cols=TARGET_COLS,
            input_feature_cols=INPUT_FEATURES,
            batch_size=BATCH_SIZE
        )



        baseline = Baseline(target_cols=TARGET_COLS)
        baseline.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )
        baseline_results = baseline.evaluate(best_window.baseline_test, verbose=0, return_dict=True)


        best_model = make_rnn(
            input_width=best_horizon_hyperparams["input_width"],
            num_features=len(INPUT_FEATURES),
            lstm_units=best_horizon_hyperparams["lstm_units"],
            dropout=best_horizon_hyperparams["dropout"],
            dense_units=best_horizon_hyperparams["dense_units"],
            num_outputs=len(TARGET_COLS),
            lr=best_horizon_hyperparams["lr"]
        )

        best_model.fit(
            best_window.train,
            epochs=EPOCHS,
            validation_data=best_window.val,
            verbose=0
        )

        test_results = best_model.evaluate(best_window.test, verbose=0, return_dict=True)

        y_t_list = []
        y_pred_baseline_list = []

        for gt_history, gt_future in best_window.baseline_test:

            y_t_list.append(gt_future.numpy())
            # Set baseline to previous output
            y_pred_baseline_list.append(gt_history[:, -1, :].numpy())

        y_t = np.concatenate(y_t_list, axis=0)
        y_pred_baseline = np.concatenate(y_pred_baseline_list, axis=0)

        y_pred_lstm_list = []

        for X_batch, _ in best_window.test:
            pred_batch = best_model.predict(X_batch, verbose=0)
            y_pred_lstm_list.append(pred_batch)

        y_pred_lstm = np.concatenate(y_pred_lstm_list, axis=0)

        plot_model_vs_baseline_vs_actual(
            y_t=y_t,
            y_pred_lstm=y_pred_lstm,
            y_pred_baseline=y_pred_baseline,
            pollutant_names=TARGET_COLS,
            horizon=horizon,
            hours=300
        )

        final_results[horizon] = {
            "baseline": baseline_results,
            "rnn": test_results,
            "best_params": best_horizon_hyperparams
        }


    print("\n\nSUMMARY (Best RNNs)")
    print("Horizon | RNN Loss | Baseline Loss | Params")

    for h in HORIZONS:
        r = final_results[h]
        print(f"{h}\t| {np.sqrt(r['rnn']['loss']):.4f} | "
              f"{np.sqrt(r['baseline']['loss']):.4f} | {r['best_params']}"
        )


if __name__ == "__main__":
    main()
