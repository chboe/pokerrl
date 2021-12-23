import csv
import os
import matplotlib.pyplot as plt

for file in os.listdir('Evaluation/'):
    with open(f'Evaluation/{file}', 'r', encoding='UTF8', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        steps = []
        winnings = []
        for i, row in enumerate(reader):
            if i == 0:
                columns = row
            else:
                steps.append(int(row[0]))
                winnings.append(float(row[1])/10000*1000)
    
    plt.clf()
    plt.figure()
    plt.plot(steps, winnings)
    plt.ticklabel_format(style='plain')
    plt.xlim(0)
    plt.axhline(0, linestyle=':')
    plt.xlabel('Gradient descent steps')
    plt.ylabel('mbb/hand')
    plt.title(file[:-4])
    plt.savefig(f'Figures/{file[:-4]}.jpeg')


