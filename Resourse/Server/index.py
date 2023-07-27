from dataCalculate import Calculate
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import xlsxwriter
import numpy as np

def CreateExcelFile(data):
    workbook = xlsxwriter.Workbook('Example3.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    column = 0 
    content = data   
    for item in content :  
        for aa in item :         
            print(aa)
        worksheet.write(column, row, item)
        print(type(item['Yp']))
        print(item['Yp'])        
        row += 1
     
    workbook.close()

data = pd.read_excel("../data/Example#1.xlsx", sheet_name=0)
CalData = Calculate(data)

# arr = CalData.to_numpy()
# CreateExcelFile(CalData['indices'])

sns.heatmap(CalData['correlations']['pearson'], annot=True)
plt.savefig('./public/img_plt/my_pearson.png')
plt.close()
# plt.show()

sns.heatmap(CalData['correlations']['spearman'], annot=True)
plt.savefig('./public/img_plt/my_spearman.png')
plt.close()
# plt.show()

plt.hist(CalData['indices'][sys.argv[1]])
plt.savefig('./public/img_plt/my_hist.png')
plt.close()
# plt.show()
print("aaa : "+ sys.argv[2])


ax = plt.figure().add_subplot(projection='3d')

colors = ('r', 'g', 'b', 'k')
np.random.seed(19680801)

x = np.random.sample(20 * len(colors))
y = np.random.sample(20 * len(colors))
c_list = []
for c in colors:
    c_list.extend([c] * 20)
# By using zdir='y', the y value of these points is fixed to the zs value 0
# and the (x, y) points are plotted on the x and z axes.
ax.scatter(x, y, zs=0, zdir='y', c=c_list, label='points in (x, z)')

# Make legend, set axes limits and labels
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('Yp')
ax.set_ylabel('Ys')
ax.set_zlabel('Z')

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=20., azim=-35, roll=0)

plt.show()



