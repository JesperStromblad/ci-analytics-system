from pymongo import MongoClient
import pandas as pd

# pprint library is used to make the output look more pretty
from pprint import pprint
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
    
    # Get list of distinct commit for the collection
    #list_commit = col.distinct("git_commit")
    list_commit = col.find().distinct("git_commit")
    # If there are less than 5 records then we append some dummy elements 
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
def get_execution_time_by_test_case(collection_name, commit, name):
    # List to store unit test data
    list_test_case_data = []

     # Collection Name
    col = db[collection_name]

    # Find documents by commit
    test_case_document=col.find({'git_commit': commit})

    # Iterate over all documents and store them to a list 
    for data in test_case_document:
        list_test_case_data.append(eval(data['unit_test_data']))
    df = pd.DataFrame([t for lst in list_test_case_data for t in lst], columns=["test_name", "execution_time", "memory"])
    return df.loc[df['test_name'] == name]


commit_list = get_last_five_commits("inputcase")
print (commit_list)

#df = get_execution_time_by_test_case('resource', 'dd0c0035', "test_to_timestamp")
