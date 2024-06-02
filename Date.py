import pandas as pd
df = pd.read_csv("C:/Users/shahn/Documents/Book12.csv")
Date = input('enter data -> dd-mm-yyyy')
df[Date] = 'A'
df.to_csv("C:/Users/shahn/Documents/Book12.csv",index = False)
df
