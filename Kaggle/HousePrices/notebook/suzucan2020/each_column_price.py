import pandas as pd

df = pd.read_csv("../../../../Kaggle/HousePrices/input/train.csv")
 
print(df)
print(df[["MSSubClass", "SalePrice"]])

for col in df.columns:
    #print(col)
    print(df.groupby(col)["SalePrice"].describe().T)

#print(df.columns)
