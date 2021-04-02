import json

import matplotlib.pyplot as plt
import seaborn as sns

total_text = []

for item in ['train', 'dev', 'test']:
    with open('cluener_public/' + item + '.json') as f:
        while True:
            line = f.readline()
            if not line:
                break
            total_text.append(json.loads(line.strip())['text'])

num = len(total_text)
print('text num:', num)

text_lens = list(map(len, total_text))
print('max len:', max(text_lens))

sns.scatterplot(x=range(num), y=text_lens)
plt.show()
