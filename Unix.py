import os
disk = os.system("df -h ") 
with open("demo.txt", 'w') as outfile:
        outfile.write(" Demo testing: %s\n" % disk)

