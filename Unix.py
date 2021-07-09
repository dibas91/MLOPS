disk = os.subprocess("df -h")
with open("evalution.txt", 'w') as outfile:
        outfile.write(" Demo testing: {%s}\n\n" % disk)

        