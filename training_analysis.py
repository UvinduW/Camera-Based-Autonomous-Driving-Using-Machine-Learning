import json
import matplotlib.pyplot as plt

# Load all the history files
test_1 = json.load(open('Test_1/steer_history_full.json', 'r'))
test_2 = json.load(open('Test_2/steer_history_full.json', 'r'))
test_3 = json.load(open('Test_3/steer_history_full.json', 'r'))
test_4 = json.load(open('Test_4/steer_history_full.json', 'r'))
test_5 = json.load(open('Test_5/steer_history_full.json', 'r'))
test_6 = json.load(open('Test_6/steer_history_full.json', 'r'))
test_7 = json.load(open('Test_7/steer_history_full.json', 'r'))
test_8 = json.load(open('Test_8/steer_history_full.json', 'r'))
test_9 = json.load(open('Test_9/steer_history_full.json', 'r'))
test_10 = json.load(open('Test_10/steer_history_full.json', 'r'))

# Difference between batch sizes
plt.title('Model Loss Evolution')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.plot(test_1['loss'][0:21])
plt.plot(test_2['loss'][0:21])
plt.legend(['Test 1 (batch size = 10)', 'Test 2 (batch size = 1000)'], loc='upper right')
plt.show()

# Performance of Network 1
plt.title('Network 1 Loss Evolution')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.plot(test_1['loss'])
plt.plot(test_2['loss'])
plt.plot(test_3['loss'])
plt.plot(test_4['loss'])
plt.legend(['Test 1', 'Test 2', 'Test 3', 'Test 4'], loc='upper right')
plt.show()

# Performance over the different networks
plt.title('Loss Evolution of all networks')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.plot(test_1['loss'])
plt.plot(test_5['loss'])
plt.plot(test_6['loss'])
plt.plot(test_7['loss'])
plt.plot(test_8['loss'])
plt.plot(test_9['loss'])
plt.plot(test_10['loss'])
plt.legend(['Test 1 (Network 1)', 'Test 5 (Network 2)', 'Test 6 (Network 3)', 'Test 7 (Network 4)', 'Test 8 (Network 5)', 'Test 9 (Network 6)', 'Test 10 (Network 7)'], loc='upper right')
plt.show()

# Performance of Test 4
plt.title('Test 4 Loss Evolution')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.plot(range(0, 100), test_4['loss'])
plt.xticks(range(0,110,10))
plt.legend(['Test 4 (Network 1 with neutral frame drop implemented)'], loc='upper right')
plt.show()
