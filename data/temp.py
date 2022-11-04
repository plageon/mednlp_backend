import csv
import json
import random

with open('分期分型.csv', 'r', encoding='gbk') as f:
    csv_reader = csv.DictReader(f)
    cancer_types = {}
    period1s = {}
    period0s = {}
    snts = []
    snt_ids = {}
    last_doc = ''
    last_snt = ''
    snt_cnt = -1
    snt_id = ''
    last_context = ''
    for row in csv_reader:
        snt_id = row['id']
        entity = row['value']
        snt = row['content']
        if snt != last_context:
            # snt_ids[snt_id] = len(snt_ids)
            snts.append({
                'id': snt_id,
                'text': snt,
                'cancer_type': '',
                'period0': '',
                'period1': ''
            })
            snt_cnt+=1
        last_context = snt
        value = row['value'].strip().split('\"')[0].split('\n')[0]
        attribute = row['attribute'].strip()
        assert attribute in ['分期', '分型']
        if attribute == '分期':
            if value == '无':
                period0 = '无'
                period1 = '无'

            else:
                if ',' in value:
                    period0, period1 = value.split(',')
                if '，' in value:
                    period0, period1 = value.split('，')
            snts[snt_cnt]['period0'] = period0
            snts[snt_cnt]['period1'] = period1
            if period0 not in period0s:
                period0s[period0] = len(period0s)
            if period1 not in period1s:
                period1s[period1] = len(period1s)
        elif attribute == '分型':
            cancer_type = value
            if cancer_type not in cancer_types:
                cancer_types[cancer_type] = len(cancer_types)

            snts[snt_cnt]['cancer_type'] = value
        # print(row)
random.shuffle(snts)
split_point = len(snts) // 10
train_data = snts[:split_point * 8]
dev_data = snts[split_point * 8:split_point * 9]
test_data = snts[split_point * 9:]
with open('period_type_data/train.json', 'w', encoding='utf-8') as f:
    for item in train_data:
       f.write(json.dumps(item, ensure_ascii=False)+'\n')
with open('period_type_data/dev.json', 'w', encoding='utf-8') as f:
    for item in dev_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
with open('period_type_data/test.json', 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('period_type_data/period0s.dict', 'w', encoding='utf-8') as f:
    f.write(json.dumps(period0s, indent=2, ensure_ascii=False))

with open('period_type_data/period1s.dict', 'w', encoding='utf-8') as f:
    f.write(json.dumps(period1s, indent=2, ensure_ascii=False))

with open('period_type_data/cancer_types.dict', 'w', encoding='utf-8') as f:
    f.write(json.dumps(cancer_types, indent=2, ensure_ascii=False))
