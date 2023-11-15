from sklearn.datasets import load_breast_cancer
import pandas as pd
import pymongo

url = r"mongodb+srv://nikhilsinghxlx:Nikhilsinghxlx@cluster0.9kjhcgg.mongodb.net/?retryWrites=true&w=majority"
DataBase_Name = "Projects" 
Collection_Name = "Breast_cancer"

if __name__ == "__main__":
    data = load_breast_cancer()

    x = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=["target"])

    # concatenating x and y for EDA 
    df = pd.concat([x, y], axis=1)

    print(f"Rows and Columns: {df.shape}")

    # connection with the database
    client = pymongo.MongoClient(url)

    # database creation
    db = client[DataBase_Name]

    # collection creation within the database
    collection = db[Collection_Name]

    # converting DataFrame to JSON format
    data_in_json = df.to_dict(orient="records")

    # insert data into the collection
    collection.insert_many(data_in_json)

    print("Data is Successfully inserted")
