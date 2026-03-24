# import gcsfs

# fs = gcsfs.GCSFileSystem(token="google_default")

# print("Exists:", fs.exists("iris-csv/data/iris.csv"))
# print("List bucket root:", fs.ls("iris-csv"))
# print("List data folder:", fs.ls("iris-csv/data"))


import gcsfs

from iris_production_project.config import RAW_PATH

fs = gcsfs.GCSFileSystem()

print("Exists:", fs.exists("iris-csv/data/iris.csv"))
print("List data:", fs.ls("iris-csv/data/"))

print (RAW_PATH)