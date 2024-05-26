import csv
import os

fields = ['img', 'label']
#have to cahnge to match your path
directory = ('C:\\Users\\maxsc\\OneDrive\\Documents\\GitHub\COMP472_Ai_Project\\Train\\angry')
#csv file to write labelled images to
filename = "labelledImages.csv"


#open csv file to write
with open(filename, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(fields)

    #iterate through files
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            writer.writerow({filename, 'angry'})



