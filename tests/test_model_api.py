import pandas as pd

user_histories_df = pd.DataFrame(
    {
        "AccountId": [
            12345,
            12345,
            12345,
            12345,
            12345,
            12345,
            12345,
            12345,
            12345,
            12345,
            12345,
        ],
        "OccurredTime": [
            "2023-10-7 13:27:00.123456",
            "2023-11-11 13:27:05.740736",
            "2023-10-8 13:27:05.740736",
            "2023-11-13 13:27:05.740736",
            "2023-10-10 13:27:05.740736",
            "2023-11-13 13:27:05.740736",
            "2023-10-5 13:27:05.740736",
            "2023-11-12 13:27:05.740736",
            "2023-12-1 13:27:01.246912",
            "2024-01-9 13:27:01.246912",
            "2024-01-10 13:27:05.740736",
        ],
        "DefinitionId": [
            "one",
            "one",
            "one",
            "two",
            "two",
            "three",
            "three",
            "three",
            "four",
            "four",
            "four",
        ],
        "ApplicationId": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
    },
    index=pd.Index([1, 1, 2, 2, 5, 5, 4, 4, 3, 3, 3], name="PersonId"),
)

user_histories_df["OccurredTime"] = pd.to_datetime(user_histories_df["OccurredTime"])
user_histories_df["DefinitionId"] = user_histories_df["DefinitionId"].astype(
    "string[pyarrow]"
)
user_histories_df["ApplicationId"] = user_histories_df["ApplicationId"].astype(
    "string[pyarrow]"
)


# def test_model_predict():
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         raw_data_path = Path(tmpdirname) / "user_histories.parquet"
#         processed_data_path = Path(tmpdirname) / "processed"

#         user_histories_df.to_parquet(raw_data_path, engine="pyarrow")

#         model = TimeSeriesDownstreamSolution(
#             raw_data_path=raw_data_path,
#             processed_data_path=processed_data_path,
#             epochs=1,
#             learning_rate=0.001,
#         )

#         model = model.train()

#         predictions = model.predict(raw_data_path)
#         assert predictions.shape == (5, 1)
#         assert predictions.index.name == "PersonId"
#         assert predictions.columns == ["prediction"]
