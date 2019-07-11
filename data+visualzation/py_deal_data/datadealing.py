import pandas as pd
from numpy import array
from math import sqrt

from pyspark.mllib.clustering import KMeans, KMeansModel

file_dir = "C:/Users/15900/Desktop/big data project/"

# with open("C:/Users/15900/Desktop/big data project/Speed Dating OCEAN1.CSV", 'rb') as f:
#   text = f.read()

'''
delete iid = 28, 58, 59, 136, 339, 340, 346 without self preference and self assessment
delete iid = 413 without date value
'''

class check_data:

    def __init__(self, file_name):

        self.file_path = file_dir
        self.file_name = file_name
        old_file_path = file_dir + file_name
        self.df = pd.read_csv(old_file_path, encoding = 'unicode_escape')

    def print_out(self,rows = None, head = 1):
        if head == 1:
            if rows is not None:
                result = self.df.head(rows)
            else:
                result = self.df.head()
        else:
            if rows is not None:
                result = self.df.tail(rows)
            else:
                result = self.df.tail()
        return result

    def check_is_null(self):
        result = self.df.isnull()
        return(result)

    def check_null_columns(self):
        result = self.df.isnull().any()
        return result

    def where_is_null(self):
        result = self.df[self.df.isnull().values==True]
        return result

    def drop_data(self, num, axis):
        # axis = 1, drop a list of columns
        # axis = 0, drop a list of rows
        self.df.drop(self.df.columns[num], axis=axis).head()

    def drop_duplicate(self, columns):
        # columns is a list of column names
        self.df.drop_duplicates(columns, 'first', True)

    def save_file(self):
        new_file_name = self.file_path + "new_" + self.file_name
        self.df.to_csv(new_file_name)

    def create_new_columns(self, new_column_name):
        self.df[new_column_name] = self.df['max_time'] - ['min_time']

    # get data of the specified rows and specified column
    def get_specified_data(self, row1, row2, col1, col2):
        result = self.df.ix[row1 : row2, col1 : col2]
        return result

    def get_column(self, columns):
        result = self.df[columns]
        return result

file_1 = "Speed Dating OCEAN.CSV"
file_2 = "owneva.csv"
file_3 = "othereva.csv"
file_4 = "node.CSV"
file_5 = "Speed Dating OCEAN2.CSV"
file_6 = "Speed Dating Data.csv"

df1 = check_data(file_1)
df2 = check_data(file_2)
df3 = check_data(file_3)
df4 = check_data(file_4)
result1 = df1.print_out()
result2 = df2.check_is_null()
result3 = df1.check_null_columns()
result4 = df3.where_is_null()
result5 = df4.drop_duplicate("iid")
df4.save_file()
print(result2)
print(result3)
print(result4)

selected_file = df1.get_column(["iid","pid","match"])
selected_file.to_csv("C:/Users/15900/Desktop/big data project/node.csv")



file = pd.read_csv("C:/Users/15900/Desktop/big data project/node.csv")
print(file[file["match"] == 1].head(5))

print(selected_file)


df1 = pd.read_csv("C:/Users/15900/Desktop/big data project/othereva.csv")
df1.reindex(columns = ["gender22"], fill_value = 0)
col_name = df1.columns.tolist()
col_name.insert(col_name.index("gender2"),'gender3')
df1.insert(4, "gender3", value = 0, )
print(df1.columns)
print(df1["gender3"].head(5))
print(df1["gender"])
    #df1["gender3"] = 0

print(df1["gender3"].head(5))
result = df1["gender"] + df1["gender22"]
result.head()
df1.to_csv("C:/Users/15900/Desktop/big data project/new.csv")


df5 = pd.read_csv(u"Speed Dating OCEAN.CSV")

df6 = pd.read_csv("Speed Dating OCEAN.CSV")

outfile = pd.merge(df5, df6, how='left', left_on=u'pid', right_on='iid')

outfile.to_csv('outfile.csv', index=False, encoding='gbk')

df7 = pd.read_csv("C:/Users/15900/Desktop/big data project/Speed Dating Data.csv", encoding = 'unicode_escape')

result6 = df7[(df7["round"] == 10)]
print(result6[["iid","pid","match"]].head(100))

result7 = df7[(df7["round"] == 21)]
print(result7[["iid","pid","match"]].head())

df7.rename(columns={"iid":"source","pid":"target"})








