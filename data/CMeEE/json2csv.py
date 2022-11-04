import json
import csv

with open('CMEE_test.json','r',encoding='utf-8') as f:
    docs=json.loads(f.read())

fields=['']
with open('CMEE_test.csv','w',encoding='utf-8') as f:
    csv_writer=csv.DictWriter(f,)
    for doc in docs:
        csv_writer.w