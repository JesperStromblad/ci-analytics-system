from pymongo import MongoClient
import pandas as pd
from functools import reduce
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, DBSCAN
from sklearn.decomposition import PCA
# pprint library is used to make the output look more pretty
from pprint import pprint








# Input-level resource dataframe columns
input_resource_columns = ["_key","avg_mem" ,"avg_time"]

# Input-level trace dataframe columns
input_trace_columns = ["_key", 'total_function_calls', 'total_no_statements_executed',"total_no_iterations",'conditional_encode_value']

# Input dataframe columns
input_columns = ["_key", "size"]


# Collection names

collection_name = ["inputcase"]



# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb://localhost:27017/admin")
# In our case, mongodb is not accessible from outside world. We can obviously set username and password
db=client.admin

# Connect to the database
db = client["ci-db"]


# Memory
memory = []

# Initialize commit memory
def init_commit_memory(list_commit):
    memory.append(list_commit[len(list_commit)-1])


# Update queue with Commit
def update_commit_memory(commit_value): 

    if commit_value != 'x': 
        if commit_value != memory[0]:
            memory.pop()
            memory.append(commit_value)


"""
Get last five commits.
"""

def get_commits ():
    return get_last_five_commits()

# Return the clicked commit
def get_last_commit():
        if memory:
            return memory
        else:
            return get_last_five_commits()[4]

# Check the content of commit memory queue
def check_commit_memory():
    return memory




# Get list of last five commit hash
def get_last_five_commits():

    # Collection Name
    col = db[collection_name[0]]
    
    # list of commit
    list_commit = []

  

    # Get list of distinct commit for the collection
    document = col.find().sort([("_id", 1)])
    for record in document:
        if record["git_commit"] not in list_commit:
            list_commit.append(record["git_commit"])



      # If commits are less than 5 then we add a negative symbol
    if len(list_commit) < 5:
        empty_commit = 5 - len(list_commit)    
        for index in range(0,empty_commit):
              list_commit.insert(index, "x")

    # Check if we have exactly five commits, we just return
    if len(list_commit) == 5:
        return list_commit

    # In case if there are more than 5, select the recent five commit
    else: 
        return list_commit[-5:]

    return list_commit      





#df = get_dataframe_unit_test_by_commit('resource', '53cf30f5', resource_columns, 'unit_test_data')

# Get execution time over time of different test cases
def get_dataframe_unit_test_by_commit(collection_name, commit, dataframe_columns=None ,only_single_test_run=False):
    
    # Test-level resource dataframe columns
    if collection_name == 'resource':
        dataframe_columns = ["test_name", "execution_time", "memory"]
    
    # List to store unit test data
    list_test_case_data = []

     # Collection Name
    col = db[collection_name]


    # Find documents by commit
    if only_single_test_run:
        test_case_document = col.find_one({'git_commit': {'$eq': commit}})
        df = pd.DataFrame(eval(test_case_document['unit_test_data']), columns=dataframe_columns)

    else:    
        test_case_document=col.find({'git_commit': {'$eq': commit}} )
        # Iterate over all documents and store them to a list 
        for data in test_case_document:
            list_test_case_data.append(eval(data['unit_test_data']))
        df = pd.DataFrame([t for lst in list_test_case_data for t in lst], columns=dataframe_columns)
    return df 

# Get either time or memory information, also one can slice dataframe based on per test case
def get_test_resource_information(df, column_name, test_name=None):

    # We can get either time or memory information
    df = df [['test_name', column_name]]

    # If we just need information about a test case
    if test_name is not None:
        return df.loc[df['test_name'] == test_name]

    return df

# Takes the average of either time or memory based on each test case
def get_resource_average(df, column_name):
    # We can get either time or memory information
    df = df [['test_name', column_name]]

    group_mem =  df.groupby('test_name', as_index=False).mean()

    return group_mem
# Get test result status
def get_test_result_status(profiling_type, commit):
    # Test-level trace dataframe columns
    trace_columns = ['test_name','test_func_calls', 'line_numbers', 'per_test_iterations', 'encode_per_test_cond', 'result']

    # Slice the dataframe to select only test name vs result for 
    return get_dataframe_unit_test_by_commit(profiling_type, commit, trace_columns, 'unit_test_data')





def check_get_input_df(collection_name, commit):
    
    # Input tuple
    list_tuple = []
    # Collection Name
    col = db[collection_name]

    # Find documents by commit
    document=col.find({'git_commit': {'$eq': commit}} )
    
    if collection_name == "inputcase":
        for data in document:
             list_tuple.append ((data["_key"], data["size"]))    

        df = pd.DataFrame([t for t in list_tuple], columns=input_columns)
    
    elif collection_name == "trace":
            for data in document:
                list_tuple.append ((data["_key"], data['total_function_calls'], data['total_no_statements_executed'], data["total_no_iterations"], data['conditional_encode_value']))    
            df = pd.DataFrame([t for t in list_tuple], columns=input_trace_columns)

    elif collection_name == "resource":    
            for data in document:
                list_tuple.append ((data["_key"], data['avg_mem'], data['avg_time']))    
            df = pd.DataFrame([t for t in list_tuple], columns=input_resource_columns)


    return df




# Merge all dataframe
def merge_dataframes(commit):
    return reduce(lambda x,y: pd.merge(x,y, on='_key', how='outer'), [check_get_input_df('inputcase', commit),
                                                                       check_get_input_df('trace', commit),
                                                                       check_get_input_df('resource', commit)
                                                                       ])

def merge_test_level_dataframes(commit):
    trace_columns = ['test_name','test_func_calls', 'line_numbers', 'per_test_iterations', 'encode_per_test_cond', 'result']
    return reduce(lambda x,y: pd.merge(x,y, on='test_name', how='outer'), [get_dataframe_unit_test_by_commit('resource', commit, True),
                                                                           get_dataframe_unit_test_by_commit('trace',commit,trace_columns, True)
    
                                                                      
                                                                       ])

#df = get_dataframe_unit_test_by_commit('resource', 'f1bc57db', resource_columns, 'unit_test_data')


def normalized_data(df, type=None):

    if type == 'incorrelation':
        columns = ['size','FunCalls', 'TExeStmt', 'TNoItr', 'ExeCond', 'mem', 'time']

    else:
        columns = ['test_func_calls', 'line_numbers', 'per_test_iterations', 'encode_per_test_cond', 'result','mem', 'time']

    from sklearn import preprocessing
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    
    if type==None:
        normal_accumulated_df = pd.DataFrame(x_scaled)
    else:
        normal_accumulated_df = pd.DataFrame(x_scaled, columns=columns)
    return normal_accumulated_df


def clustering(clustering_algo,df, cluster_value):
    if clustering_algo == 'kmeans':
        kmeans = KMeans(n_clusters = cluster_value, init="k-means++", max_iter = 500, n_init = 10, random_state = 123)
        identified_clusters = kmeans.fit_predict(df)
        
    elif clustering_algo == 'agglomerative':    
        hc = AgglomerativeClustering(n_clusters = cluster_value, affinity = "euclidean", linkage = "ward")
        identified_clusters = hc.fit_predict(df)

    elif clustering_algo == 'affinitypropagation':
        hc = AffinityPropagation()
        identified_clusters = hc.fit_predict(df)

    elif clustering_algo == 'dbscan':
        DBS = DBSCAN(eps = 0.2, min_samples = 40)
        identified_clusters = DBS.fit_predict(df)

    elif clustering_algo == 'gmm':
        from sklearn import mixture
        gmm = mixture.GaussianMixture(n_components=cluster_value, covariance_type='full').fit(df)
        identified_clusters = gmm.predict(df)

    
    df['Cluster'] = identified_clusters
    return df


## Usage of methods

# Get list of commit
#commit_list = get_last_five_commits()


# Get execution time over different inputs for each test case
#df = get_dataframe_unit_test_by_commit('resource', '53cf30f5', resource_columns, 'unit_test_data')


# Get dataframe by resource
#df = get_test_resource_information(df, "execution_time", "test_to_datetime")

# Get average of resource by test case
#df = get_resource_average(df, "memory")

# Test Case 
#df = get_test_result_status('trace', '617fddea', trace_columns, 'unit_test_data')


# Merge dataframes
# df = merge_dataframes('617fddea')


# # Merge test-level dataframe
# #df = merge_test_level_dataframes('617fddea')
# #df = df.loc[:, df.columns != 'test_name']

# df = normalized_data(df)

# df = clustering('kmeans', df)

