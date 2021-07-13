from pymongo import MongoClient
import pandas as pd

# pprint library is used to make the output look more pretty
from pprint import pprint


# Resource dataframe columns
resource_columns = ["test_name", "execution_time", "memory"]


# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb://localhost:27017/admin")
# mongodb is not accessible from outside because scripts are run on the server. We can obviously set username and password
db=client.admin

# Connect to the database
db = client["ci-db"]
    

# Get list of last five commit hash
def get_last_five_commits(collection_name):
    # Collection Name
    col = db[collection_name]
    
    # list of commit
    list_commit = []

    # Get list of distinct commit for the collection
    document = col.find().sort([("_id", 1)])
    for record in document:
        if record["git_commit"] not in list_commit:
            list_commit.append(record["git_commit"])
    
    #If there are less than 5 records then we append some dummy elements 
    if len(list_commit) < 5:
        empty_commit = 5 - len(list_commit)    
        for index in range(0,empty_commit):
              list_commit.insert(index, "empty")

    # Check if we have exactly five commits, we just return
    elif len(list_commit) == 5:
        return list_commit

    # In case if there are more than 5, select the recent five commit
    else: 
        return list_commit[-5:]

    return list_commit      

# Get execution time over time of different test cases
def get_dataframe_by_commit(collection_name, commit):
    # List to store unit test data
    list_test_case_data = []

     # Collection Name
    col = db[collection_name]

    # Find documents by commit
    test_case_document=col.find({'git_commit': {'$eq': commit}} )

    # Iterate over all documents and store them to a list 
    for data in test_case_document:
        list_test_case_data.append(eval(data['unit_test_data']))
    df = pd.DataFrame([t for lst in list_test_case_data for t in lst], columns=resource_columns)
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

    return df.groupby('test_name', as_index=False).mean()

## Usage of methods

# Get list of commit
commit_list = get_last_five_commits("inputcase")

# Get execution time over different inputs for each test case
df = get_dataframe_by_commit('resource', 'f1bc57db')


# Get dataframe by resource
#df = get_test_resource_information(df, "execution_time", "test_to_datetime")

# Get average of resource by test case
df = get_resource_average(df, "memory")
print (df)

# for test in df['test_name'].unique():
#     print(df.loc[df['test_name'] == test])