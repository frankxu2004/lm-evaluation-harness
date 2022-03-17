import csv
import os
import json
from collections import Counter

# total_count = 0
# c = Counter()
# with open('/home/fangzhex/md/projects/github_repositories.csv', newline='') as csv_f:
#     reader = csv.reader(csv_f)
#     for row in reader:
#         total_count += 1
#         c[row[-1]] +=1

# for lang in c:
#     print(lang, c[lang]/total_count * 95.16 )
# exit()

pile_repos = set()
with open('/home/fangzhex/md/projects/github_repositories.csv', newline='') as csv_f:
    reader = csv.reader(csv_f)
    for row in reader:
        pile_repos.add(row[0])



test_repos = set()
for root, _, files in os.walk('/data/vincent/Mining/CodeData/DataCollection/Mining/Code'):
    for file in files:
        test_repos.add('/'.join(os.path.join(root, file).split('/')[9:11]))

print(len(test_repos))
overlapped = pile_repos.intersection(test_repos)
print(len(overlapped))

for x in overlapped:
    print(x)

json.dump(list(overlapped), open('overlapped_repos.json', 'w'))

