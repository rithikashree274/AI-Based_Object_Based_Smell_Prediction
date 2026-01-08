import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_smell_model():

    df = pd.read_csv("object_smell_high_accuracy_dataset.csv")

    print("Dataset Columns:", df.columns.tolist())

    le_object = LabelEncoder()
    le_subtype = LabelEncoder()
    le_smell = LabelEncoder()
    le_chemical = LabelEncoder()
    le_chemical_type = LabelEncoder()

    df["object_enc"] = le_object.fit_transform(df["object"])
    df["sub_type_enc"] = le_subtype.fit_transform(df["sub_type"])
    df["smell_enc"] = le_smell.fit_transform(df["smell"])
    df["chemical_enc"] = le_chemical.fit_transform(df["chemical"])
    df["chemical_type_enc"] = le_chemical_type.fit_transform(df["chemical_type"])

    X = df[["object_enc", "sub_type_enc", "intensity"]]
    y = df["smell_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nOverall Model Accuracy: {accuracy * 100:.2f}%")

    df["Predicted_Smell"] = le_smell.inverse_transform(
        model.predict(X)
    )

    output = df[
        ["object", "intensity", "Predicted_Smell", "chemical", "chemical_type"]
    ].rename(columns={
        "object": "Object_Detected",
        "intensity": "Intensity",
        "chemical": "Chemical_Name",
        "chemical_type": "Chemical_Type"
    })

    print("\n🔹 Predictions (First 20 Rows):")
    print(output.head(20))

    output.to_csv("predicted_smells.csv", index=False)
    print("\n File saved successfully: predicted_smells.csv")
