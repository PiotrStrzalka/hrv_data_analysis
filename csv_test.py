import pandas
import numpy as np

# path = "C:\\PROJEKTY\\sandbox\\hrv_data_analysis\\example_videos\\training_data.csv"

# df = pandas.read_csv(path)
# print(df)

a = np.array([111,222,333], ndmin=2)
ap = np.transpose(a)
ar = np.array(([1,2,3],[4,5,6],[7,8,9]))

arr = np.array(([11,12,13], [14,15,16], [17,18,19]))


print(a)
print(a.shape)
print(ar)
print(ar.shape)
# aarr = np.stack((ap, ar))
aar = np.concatenate((ap,ar), axis=1)

aararr = np.concatenate((aar,arr), axis = 1)

# print(aar)
print(aararr)
np.savetxt("numpytxt.txt", aararr, 
           header = "time_delta, mean_r, mean_g, mean_b, mean_h, mean_s, mean_v")

g = np.loadtxt("numpytxt.txt")

print(g.shape)
print(g[:,-1])

np.savetxt("C:\\PROJEKTY\\sandbox\\hrv_data_analysis\\numpytxt2.txt", g)
# adf = pandas.DataFrame(aararr)
# adf.to_csv("a_my_file.csv", index = False)

# aarrdf = pandas.DataFrame(aarr)
# aarrdf.to_csv("aarr_my_file.csv", index = False)

# ardf = pandas.DataFrame(aararr,
#         columns=["time_delta", "mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v"])
# ardf.to_csv("my_file.csv", index = False)


# rdf = pandas.read_csv("my_file.csv")

# print(rdf.columns)
