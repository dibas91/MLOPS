import os
disk = os.subprocess("df -h")
with open("demo.txt", 'w') as outfile:
        outfile.write(" Demo testing: {%s}\n\n" % disk)

