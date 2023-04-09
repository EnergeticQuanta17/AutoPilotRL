from dask.distributed import Client, LocalCluster

cluster = LocalCluster()
client = Client(cluster)

# use the cluster and client

cluster.close()