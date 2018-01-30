from matplotlib import pyplot as plt

picdata_file = open("../output/routine/price_voltree_slices/slice40.csv", 'r')
picdata_file.readline()  # headers, skipping

for line in picdata_file:
    if line.startswith("moment 9"):
        moment_data = (line.split(":")[1].split(';'))  # selecting data
        moment_data.pop()  # deleting new string character
        plt.plot(moment_data)
plt.show()
plt.close()
