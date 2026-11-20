import csv
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ONE_HOUR = timedelta(hours=1)
TRAIN_START = datetime(2023, 1, 1)
EVALUATION_START = datetime(2024, 12, 1)
EVALUATION_END = datetime(2025, 1, 1)
FEATURE_COLUMNS = (
    "wind_speed_ms",
    "temperature_C",
    "humidity_pct",
    "wind_speed_ms_lag_1h",
    "wind_speed_ms_lag_2h",
)
PREDICTION_COLUMNS = (
    "prediction_time",
    "target_time",
    "actual_wind_speed_ms",
    "actual_label",
    "current_wind_speed_ms",
    "baseline_always_0",
    "baseline_current_wind_ge_4",
    "high_wind_probability",
    "logistic_regression_prediction",
    "logistic_regression_prediction_0_20",
)
SCORE_COLUMNS = (
    "method",
    "evaluation_rows",
    "same_evaluation_rows",
    "actual_high_wind",
    "hits",
    "misses",
    "false_alarms",
    "correct_negatives",
    "accuracy",
    "high_wind_precision",
    "high_wind_recall",
)


def parse_datetime(value: str, column: str, row_number: int) -> datetime:
    try:
        return datetime.strptime(value, DATETIME_FORMAT)
    except ValueError as error:
        raise ValueError(
            f"Invalid {column} at CSV row {row_number}: {value!r}"
        ) from error


def main() -> None:
    csv_path = Path(__file__).with_name("wind_dataset.csv")
    required_columns = {
        "prediction_time",
        "target_time",
        "target_wind_speed_ms",
        *FEATURE_COLUMNS,
    }

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required CSV columns: {sorted(missing_columns)}")

        rows = []
        previous_target_time = None
        one_hour_prediction_offsets = 0
        non_hourly_target_intervals = []

        for row_number, row in enumerate(reader, start=2):
            prediction_time = parse_datetime(
                row["prediction_time"], "prediction_time", row_number
            )
            target_time = parse_datetime(row["target_time"], "target_time", row_number)
            if target_time - prediction_time == ONE_HOUR:
                one_hour_prediction_offsets += 1
            else:
                raise ValueError(
                    f"prediction_time and target_time are not one hour apart "
                    f"at CSV row {row_number}: {row['prediction_time']} -> "
                    f"{row['target_time']}"
                )

            if previous_target_time is not None:
                interval = target_time - previous_target_time
                if interval <= timedelta(0):
                    raise ValueError(
                        f"target_time is not strictly chronological at CSV row "
                        f"{row_number}: {row['target_time']}"
                    )
                if interval != ONE_HOUR:
                    non_hourly_target_intervals.append(
                        (previous_target_time, target_time, interval)
                    )
            previous_target_time = target_time

            wind_next_1h = float(row["target_wind_speed_ms"])
            row["label"] = 1 if wind_next_1h >= 4.0 else 0
            row["_target_datetime"] = target_time
            rows.append(row)

    training_rows = [
        row for row in rows
        if TRAIN_START <= row["_target_datetime"] < EVALUATION_START
    ]
    evaluation_rows = [
        row for row in rows
        if EVALUATION_START <= row["_target_datetime"] < EVALUATION_END
    ]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight=None, max_iter=1000),
    )
    train_features = [
        [float(row[column]) for column in FEATURE_COLUMNS]
        for row in training_rows
    ]
    train_labels = [int(row["label"]) for row in training_rows]
    model.fit(train_features, train_labels)

    evaluation_features = [
        [float(row[column]) for column in FEATURE_COLUMNS]
        for row in evaluation_rows
    ]
    high_wind_probabilities = model.predict_proba(evaluation_features)[:, 1]
    predictions_path = Path(__file__).with_name("classification_predictions.csv")
    prediction_records = []
    for row, probability in zip(
        evaluation_rows, high_wind_probabilities, strict=True
    ):
        current_wind_speed = float(row["wind_speed_ms"])
        prediction_records.append(
            {
                "prediction_time": row["prediction_time"],
                "target_time": row["target_time"],
                "actual_wind_speed_ms": row["target_wind_speed_ms"],
                "actual_label": int(row["label"]),
                "current_wind_speed_ms": row["wind_speed_ms"],
                "baseline_always_0": 0,
                "baseline_current_wind_ge_4": int(current_wind_speed >= 4.0),
                "high_wind_probability": f"{probability:.10f}",
                "logistic_regression_prediction": int(probability >= 0.50),
                "logistic_regression_prediction_0_20": int(probability >= 0.20),
            }
        )

    with predictions_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(prediction_records)

    method_columns = (
        ("항상 0", "baseline_always_0"),
        ("현재 풍속 >= 4.0 m/s", "baseline_current_wind_ge_4"),
        ("로지스틱 회귀 (확률 >= 0.50)", "logistic_regression_prediction"),
        ("로지스틱 회귀 (확률 >= 0.20)", "logistic_regression_prediction_0_20"),
    )
    evaluation_keys = [
        (record["prediction_time"], record["target_time"], record["actual_label"])
        for record in prediction_records
    ]
    score_rows = []
    for method_name, prediction_column in method_columns:
        hits = misses = false_alarms = correct_negatives = 0
        method_keys = []
        for record in prediction_records:
            actual = record["actual_label"]
            predicted = record[prediction_column]
            method_keys.append(
                (record["prediction_time"], record["target_time"], actual)
            )
            if actual == 1 and predicted == 1:
                hits += 1
            elif actual == 1 and predicted == 0:
                misses += 1
            elif actual == 0 and predicted == 1:
                false_alarms += 1
            else:
                correct_negatives += 1

        total_rows = len(prediction_records)
        actual_high_wind = hits + misses
        predicted_high_wind = hits + false_alarms
        score_rows.append(
            {
                "method": method_name,
                "evaluation_rows": total_rows,
                "same_evaluation_rows": method_keys == evaluation_keys,
                "actual_high_wind": actual_high_wind,
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_negatives": correct_negatives,
                "accuracy": f"{(hits + correct_negatives) / total_rows:.6f}",
                "high_wind_precision": (
                    f"{hits / predicted_high_wind:.6f}"
                    if predicted_high_wind
                    else "해당 없음"
                ),
                "high_wind_recall": f"{hits / actual_high_wind:.6f}",
            }
        )

    if not score_rows or not all(row["same_evaluation_rows"] for row in score_rows):
        raise ValueError("Methods were not evaluated on identical rows")

    scores_path = Path(__file__).with_name("classification_scores.csv")
    with scores_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(score_rows)

    def print_period(name: str, period_rows: list[dict[str, object]]) -> None:
        count_0 = sum(row["label"] == 0 for row in period_rows)
        count_1 = sum(row["label"] == 1 for row in period_rows)
        print(f"{name}: 전체 {len(period_rows)}행, 0={count_0}개, 1={count_1}개")

    print("시각 기준: KST (시간대 변환 없음)")
    print(f"CSV 전체 행: {len(rows)}")
    print(f"정답 시각 기준 각 행의 1시간 차이 확인: {one_hour_prediction_offsets}/{len(rows)}")
    print(f"정답 시각 시간순 유지 확인: 예 (CSV 입력 순서)")
    print(
        "인접 행의 정답 시각이 정확히 1시간 차이: "
        f"{len(rows) - 1 - len(non_hourly_target_intervals)}/{max(len(rows) - 1, 0)}"
    )
    print(f"인접 행 사이의 시간 간격 예외: {len(non_hourly_target_intervals)}건")
    for previous, current, interval in non_hourly_target_intervals[:5]:
        print(f"  {previous} -> {current} ({interval})")

    print_period("학습 (2023-01 ~ 2024-11)", training_rows)
    print_period("평가 (2024-12-01 ~ 2024-12-31)", evaluation_rows)
    print(f"분류 모델: 학습 행만 StandardScaler 표준화 후 LogisticRegression 학습")
    print("클래스 가중치: 사용 안 함")
    print(f"평가 예측 저장: {predictions_path.name} ({len(evaluation_rows)}행)")
    print(f"평가 행 동일성: 방법별 {len(evaluation_keys)}행, 모두 동일")
    print(f"실제 강풍 행: {score_rows[0]['actual_high_wind']}")
    print(f"비교 점수 저장: {scores_path.name}")
    threshold_scores = [
        score
        for score in score_rows
        if score["method"] in (
            "로지스틱 회귀 (확률 >= 0.50)",
            "로지스틱 회귀 (확률 >= 0.20)",
        )
    ]
    print("같은 모델·같은 강풍 정답 기준 (평가 744행, 실제 강풍 39행)")
    print("확률 임계값 | 적중 | 놓침 | 오경보 | 정밀도 | 재현율")
    for score in threshold_scores:
        print(
            f"{score['method']} | {score['hits']} | {score['misses']} | "
            f"{score['false_alarms']} | {float(score['high_wind_precision']):.4f} | "
            f"{float(score['high_wind_recall']):.4f}"
        )
    print(
        "방법 | 적중 | 놓침 | 오경보 | 정상판정 | 정확도 | "
        "강풍 정밀도 | 강풍 재현율"
    )
    for score in score_rows:
        print(
            f"{score['method']} | {score['hits']} | {score['misses']} | "
            f"{score['false_alarms']} | {score['correct_negatives']} | "
            f"{float(score['accuracy']):.4f} | {score['high_wind_precision']} | "
            f"{float(score['high_wind_recall']):.4f}"
        )

    if not evaluation_rows:
        print("첫 평가 행: 없음")
    else:
        first_evaluation_row = {
            key: value
            for key, value in evaluation_rows[0].items()
            if key != "_target_datetime"
        }
        print(f"첫 평가 행: {first_evaluation_row}")


if __name__ == "__main__":
    main()
