import csv
import os
import matplotlib.pyplot as plt

for file in ['model_id=2000_Network=Q_versus=Call_Agent.csv', 'model_id=2000_Network=Q_versus=Itself.csv', 'model_id=2000_Network=Q_versus=Raise_Agent.csv']:
    steps = []
    winnings = []
    columns = None
    
    with open(f'Evaluation/{file}', 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                columns = row
            else:
                steps.append(int(row[0]))
                winnings.append(float(row[1]))
    
    with open(f'Evaluation/{file}x', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['steps', 'winnings'])
        for i in range(len(steps)):
            writer.writerow([steps[i], winnings[i]/2.5])