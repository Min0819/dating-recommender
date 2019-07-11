#./spark/bin/pyspark --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11
import findspark

findspark.init()

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

conf = SparkConf()
conf.setAppName("visulization")
sc = SparkContext(conf=conf)
sc.setCheckpointDir("./checkpoint")
spark = SparkSession.builder.appName("Graph").getOrCreate()

vfile = "/home/mf3200/Bigdataproject/node_ori1.csv"
efile = "/home/mf3200/Bigdataproject/edge_ori1.csv"
V = spark.read.csv(vfile)
E = spark.read.csv(efile)
v = V.select(V._c2, V._c3).selectExpr("_c2 as id", "_c3 as wave")
e = E.select(E._c2, E._c3, E._c4, E._c5).selectExpr("_c2 as src", "_c3 as dst", "_c4 as match", "_c5 as wave")
v.show()
e.show()

from graphframes import *

g = GraphFrame(v, e)

g.vertices.show()
g.edges.show()

e = g.edges.filter("match > 0")
g = GraphFrame(v, e)

results_PR = g.pageRank(resetProbability=0.15, tol=0.01)

results_PR.vertices.select("id", "pagerank", "wave").show()
results_PR.edges.select("src", "dst", "match", "wave").show()

# result_CC = g.connectedComponents()
# result_CC.select("id", "component").orderBy("component").show()

# results_TC = g.triangleCount()
# results_TC.select("id","count").show()

result1 = results_PR.vertices.select("id", "PageRank", "wave")
# result2 = result_CC.select("id","Component")
import pandas as pd

result1 = result1.toPandas()
# result2 = result2.toPandas()
# node= pd.merge(result1, result2, on = "id")
node = result1
node = node.rename(columns={"id": "ID"})
node["ID"] = node['ID'].astype(int)
node = node.sort_values(by='ID')
node.head(10)

edge = e.select("src", "dst")
edge = edge.toPandas()
edge = edge.rename(columns={"src": "source", "dst": "target"})
edge.head(10)
node.to_csv("/home/mf3200/Bigdataproject/node_ori.csv")
edge.to_csv("/home/mf3200/Bigdataproject/edge_ori.csv")

#python - m http.server 5000