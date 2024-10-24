import csv
import time
import requests
import pytest

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from application import application

@pytest.fixture
def client():
    app = application
    app.config["TESTING"] = True

    with app.app_context():
        yield app.test_client()

test_cases = [
    {"text": "This is fake news", "expected": 'FAKE'},
    {"text": "People live on Mars", "expected": 'FAKE'},
    {"text": "The year is 2024", "expected": 'REAL'},
    {"text": "Trump was the president of United States.", "expected": 'REAL'},
]

### Predictions Test

@pytest.mark.parametrize("test_case", test_cases)
def test_prediction(client, test_case):
    test = {'text': [test_case['text']]}
    rsp = client.post('/predict', json=test)

    print(f"Test case: {test_case}")
    assert rsp.status_code == 200
    rsp = rsp.get_json()
    assert rsp['prediction'][0] == test_case['expected']


### Latency Test

ELASTIC_URL = 'http://fake-news-app-env.eba-5q3m6xxq.ca-central-1.elasticbeanstalk.com'
CSV = 'latency_results.csv'
PLOT = 'box_plot.png'

def get_plot():
    print("Starting to generate the plot...")
    data = pd.read_csv(CSV)

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Test Case #', y='Response Time (seconds)', data=data)

    plt.title('Response Time for Each Test Case')
    plt.xlabel('Test Case')
    plt.ylabel('Response Time (seconds)')
    plt.savefig(PLOT)
    plt.savefig(PLOT)
    plt.close()

    return data.groupby('Test Case #')['Response Time (seconds)'].mean()

def test_latency():
    with open(CSV, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Test Case #', 'Status', 'Timestamp', 'Response Time (seconds)'])

        for test_case_num, test_case in enumerate(test_cases):
            for i in range(100):
                start = time.time()
                rsp = requests.post(ELASTIC_URL, json={'text': [test_case['text']]})
                end = time.time()

                rsp_time = end - start

                writer.writerow(
                    [f"Test Case {test_case_num + 1}",
                     rsp,
                     time.strftime('%Y-%m-%d %H:%M:%S'),
                     rsp_time])
        
        average_latency = get_plot()
        writer.writerow(['Average', average_latency])

test_latency()