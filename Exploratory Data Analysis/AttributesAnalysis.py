# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
from pandas import DataFrame


# Set data
#-------preference----------------------
pref = DataFrame({
    'gender':['0','1'],
    'Attractive':[18.055224,26.921689],
    'Sincere':[18.305008, 16.498436],
    'Intelligent':[21.002502,19.537374],
    'Fun':[17.147292, 17.763893],
    'Ambitious':[12.827222,8.552829],
    'Interest':[12.704194, 10.996574]
})


#----------female self measure up-----------------
self_o_measure_f =  DataFrame({
    's_o':['s','o'],
    'Attractive':[7.219092,6.461401],
    'Sincere':[8.458343, 7.251053],
    'Intelligent':[8.320622,7.291202],
    'Fun':[7.893612, 6.520164],
    'Ambitious':[7.632499,6.604591]
})


#----------male self measure up-----------------
self_o_measure_m =  DataFrame({
    's_o':['s','o'],
    'Attractive':[6.951636,5.919422],
    'Sincere':[8.133061, 7.099778],
    'Intelligent':[8.486526,7.447362],
    'Fun':[7.517084, 6.280555],
    'Ambitious':[7.524783,6.952773]
})



#----------female decision rating VS selfstated preference-----------------

decisionVSpref_f = DataFrame({
    'd_p':['d','p'],
    'Attractive':[26.10708904, 18.055224],
    'Sincere':[9.12608432, 18.305008],
    'Intelligent':[9.78653768, 21.002502],
    'Fun':[23.11648473, 17.147292],
    'Ambitious':[9.21428016, 12.827222],
    'Interest':[22.64952407, 12.704194]
})



#----------male decision rating VS selfstated preference-----------------

decisionVSpref_m = DataFrame({
    'd_p':['d','p'],
    'Attractive': [26.6596794, 26.921689],
    'Sincere': [6.63116443, 16.498436],
    'Intelligent': [11.39086349, 19.537374],
    'Fun': [21.11363779, 17.763893],
    'Ambitious': [15.47184769, 8.552829],
    'Interest': [18.7328072, 10.996574]
})

df = decisionVSpref_m

# ------- PART 1: Create background

# number of variable
categories = list(df)[1:]
N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
plt.ylim(0, 40)

# plt.yticks([3, 6, 9], ["3", "6", "9"], color="grey", size=7)
# plt.ylim(0, 10)

# ------- PART 2: Add plots

# Plot each individual = each line of the data

# Ind1
values = df.loc[0].drop('d_p').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values,linewidth=0.5, linestyle='solid', color='b',label="Actual Decision")
ax.fill(angles, values, 'b', alpha=0.2)

# Ind2
values = df.loc[1].drop('d_p').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=0.5, linestyle='solid',color='g', label="Stated Preference")
ax.fill(angles, values, 'g', alpha=0.2)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.savefig('./images/selfstated_prefVSdecision_m.png')
plt.show()
