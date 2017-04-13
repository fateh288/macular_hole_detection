import xml.etree.ElementTree as ET
import glob
import numpy as np

def getCoordinatesArray(xml_file_path,xmax=2588,ymax=1958):
    train_label=[]
    for filename in glob.glob(xml_file_path+ '/*.xml'):
        tree = ET.parse(filename)
        root = tree.getroot()
        sub_label=[float(root[6][4][0].text)/xmax,float(root[6][4][1].text)/ymax,float(root[6][4][2].text)/xmax,float(root[6][4][3].text)/ymax]#(xmin,ymin,xmax,ymax)
        train_label.append(sub_label)
    return np.array(train_label)

#train_label=getCoordinatesArray('/media/fateh/01D2023161FD29C0/macular_hole/main_data/All_KMC/xml')
#for i in range(0,10):
#    print(train_label[i])

