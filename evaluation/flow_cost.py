import pickle
import matplotlib.pyplot as plt
# import cPickle as pickle
from matplotlib import legend_handler
import numpy as np

# rsfca_line4_list = pickle.load(open("rsfca_box4.pickle", "rb"))

path = "/home/test/00-fast-journal/moop/fast-integration/fast-simulation/evaluation/results/3-ary-official/"

# =======================================================================================
opt_total_cost_result = pickle.load(open(path + "opt_total_cost_result.pickle", "rb"))
opt_total_cost_err = pickle.load(open(path + "opt_total_cost_err.pickle", "rb"))

opt_comm_cost_result = pickle.load(open(path + "opt_comm_cost_result.pickle", "rb"))
opt_comm_cost_err = pickle.load(open(path + "opt_comm_cost_err.pickle", "rb"))

print "Opt:", opt_comm_cost_result

opt_buff_cost_result = pickle.load(open(path + "opt_buff_cost_result.pickle", "rb"))
opt_buff_cost_err = pickle.load(open(path + "opt_buff_cost_err.pickle", "rb"))

opt_mig_time_result = pickle.load(open(path + "opt_mig_time_result.pickle", "rb"))
opt_mig_time_err = pickle.load(open(path + "opt_mig_time_err.pickle", "rb"))

opt_exec_time_result = pickle.load(open(path + "opt_exec_time_result.pickle", "rb"))
opt_exec_time_err = pickle.load(open(path + "opt_exec_time_err.pickle", "rb"))

opt_comp_cost_result = pickle.load(open(path + "opt_comp_cost_result.pickle", "rb"))
opt_comp_cost_err = pickle.load(open(path + "opt_comp_cost_err.pickle", "rb"))

# =======================================================================================
bast_total_cost_result = pickle.load(open(path + "bast_total_cost_result.pickle", "rb"))
bast_total_cost_err = pickle.load(open(path + "bast_total_cost_err.pickle", "rb"))

bast_comm_cost_result = pickle.load(open(path + "bast_comm_cost_result.pickle", "rb"))
bast_comm_cost_err = pickle.load(open(path + "bast_comm_cost_err.pickle", "rb"))

print "CAST:", bast_comm_cost_result

bast_buff_cost_result = pickle.load(open(path + "bast_buff_cost_result.pickle", "rb"))
bast_buff_cost_err = pickle.load(open(path + "bast_buff_cost_err.pickle", "rb"))

bast_mig_time_result = pickle.load(open(path + "bast_mig_time_result.pickle", "rb"))
bast_mig_time_err = pickle.load(open(path + "bast_mig_time_err.pickle", "rb"))

bast_exec_time_result = pickle.load(open(path + "bast_exec_time_result.pickle", "rb"))
bast_exec_time_err = pickle.load(open(path + "bast_exec_time_err.pickle", "rb"))

bast_comp_cost_result = pickle.load(open(path + "bast_comp_cost_result.pickle", "rb"))
bast_comp_cost_err = pickle.load(open(path + "bast_comp_cost_err.pickle", "rb"))

# ==========================================================================================
last_total_cost_result = pickle.load(open(path + "last_total_cost_result.pickle", "rb"))
last_total_cost_err = pickle.load(open(path + "last_total_cost_err.pickle", "rb"))

last_comm_cost_result = pickle.load(open(path + "last_comm_cost_result.pickle", "rb"))
last_comm_cost_err = pickle.load(open(path + "last_comm_cost_err.pickle", "rb"))

print "LAST:", last_comm_cost_result

last_buff_cost_result = pickle.load(open(path + "last_buff_cost_result.pickle", "rb"))
last_buff_cost_err = pickle.load(open(path + "last_buff_cost_err.pickle", "rb"))

last_mig_time_result = pickle.load(open(path + "last_mig_time_result.pickle", "rb"))
last_mig_time_err = pickle.load(open(path + "last_mig_time_err.pickle", "rb"))

last_exec_time_result = pickle.load(open(path + "last_exec_time_result.pickle", "rb"))
last_exec_time_err = pickle.load(open(path + "last_exec_time_err.pickle", "rb"))

last_comp_cost_result = pickle.load(open(path + "last_comp_cost_result.pickle", "rb"))
last_comp_cost_err = pickle.load(open(path + "last_comp_cost_err.pickle", "rb"))

# ==========================================================================================
rast_total_cost_result = pickle.load(open(path + "rast_total_cost_result.pickle", "rb"))
rast_total_cost_err = pickle.load(open(path + "rast_total_cost_err.pickle", "rb"))

rast_comm_cost_result = pickle.load(open(path + "rast_comm_cost_result.pickle", "rb"))
rast_comm_cost_err = pickle.load(open(path + "rast_comm_cost_err.pickle", "rb"))

print "RAST:", rast_comm_cost_result

rast_buff_cost_result = pickle.load(open(path + "rast_buff_cost_result.pickle", "rb"))
rast_buff_cost_err = pickle.load(open(path + "rast_buff_cost_err.pickle", "rb"))

rast_mig_time_result = pickle.load(open(path + "rast_mig_time_result.pickle", "rb"))
rast_mig_time_err = pickle.load(open(path + "rast_mig_time_err.pickle", "rb"))

rast_exec_time_result = pickle.load(open(path + "rast_exec_time_result.pickle", "rb"))
rast_exec_time_err = pickle.load(open(path + "rast_exec_time_err.pickle", "rb"))

rast_comp_cost_result = pickle.load(open(path + "rast_comp_cost_result.pickle", "rb"))
rast_comp_cost_err = pickle.load(open(path + "rast_comp_cost_err.pickle", "rb"))

# ==========================================================================================
base_total_cost_result = pickle.load(open(path + "base_total_cost_result.pickle", "rb"))
base_total_cost_err = pickle.load(open(path + "base_total_cost_err.pickle", "rb"))

base_comm_cost_result = pickle.load(open(path + "base_comm_cost_result.pickle", "rb"))
base_comm_cost_err = pickle.load(open(path + "base_comm_cost_err.pickle", "rb"))

print "BASE:", base_comm_cost_result

base_buff_cost_result = pickle.load(open(path + "base_buff_cost_result.pickle", "rb"))
base_buff_cost_err = pickle.load(open(path + "base_buff_cost_err.pickle", "rb"))

base_exec_time_result = pickle.load(open(path + "base_exec_time_result.pickle", "rb"))
base_exec_time_err = pickle.load(open(path + "base_exec_time_err.pickle", "rb"))

# ==========================================================================================

past_total_cost_result = pickle.load(open(path + "past_total_cost_result.pickle", "rb"))
past_total_cost_err = pickle.load(open(path + "past_total_cost_err.pickle", "rb"))

past_comm_cost_result = pickle.load(open(path + "past_comm_cost_result.pickle", "rb"))
past_comm_cost_err = pickle.load(open(path + "past_comm_cost_err.pickle", "rb"))

print "PAST:", past_comm_cost_result

past_buff_cost_result = pickle.load(open(path + "past_buff_cost_result.pickle", "rb"))
past_buff_cost_err = pickle.load(open(path + "past_buff_cost_err.pickle", "rb"))

past_mig_time_result = pickle.load(open(path + "past_mig_time_result.pickle", "rb"))
past_mig_time_err = pickle.load(open(path + "past_mig_time_err.pickle", "rb"))

past_exec_time_result = pickle.load(open(path + "past_exec_time_result.pickle", "rb"))
past_exec_time_err = pickle.load(open(path + "past_exec_time_err.pickle", "rb"))

past_comp_cost_result = pickle.load(open(path + "past_comp_cost_result.pickle", "rb"))
past_comp_cost_err = pickle.load(open(path + "past_comp_cost_err.pickle", "rb"))

# -----------------------------------------------------

fig = plt.figure(1)
ax1 = fig.add_subplot(111)

# flierprops = dict(marker='o', markerfacecolor='green', markersize=12,linestyle='none')
# medianprops = dict(linestyle='-.', linewidth=1.5, color='firebrick')

colors = ['darkorange', 'green', 'firebrick', 'navy', 'blue', 'purple']

# markeredgewidth=2,
index = np.array([1, 2, 3, 4, 5])
fig1 = plt.errorbar(index, past_total_cost_result, yerr=past_total_cost_err, color=colors[5], linewidth=2, marker='x', markersize=15, label=r'Tabu', markeredgewidth=2, capsize=3, fillstyle='none')
fig2 = plt.errorbar(index, rast_total_cost_result, yerr=rast_total_cost_err, color=colors[4], linewidth=2, marker='^', markersize=8, label=r'Random', capsize=3, fillstyle='none')
fig3 = plt.errorbar(index, last_total_cost_result, yerr=last_total_cost_err, color=colors[3], linewidth=2, marker='o', markersize=8, label=r'LAST', capsize=3)
fig4 = plt.errorbar(index, base_total_cost_result, yerr=base_total_cost_err, color=colors[2], linestyle='--', linewidth=2, marker='.', markersize=5, label=r'No migration', capsize=3, fillstyle='none')
fig5 = plt.errorbar(index, bast_total_cost_result, yerr=bast_total_cost_err, color=colors[1], linewidth=2, marker='s', markersize=10, label=r'CAST', capsize=3, fillstyle='none')
fig6 = plt.errorbar(index, opt_total_cost_result, yerr=opt_total_cost_err, color=colors[0], linewidth=2, marker='.', label=r'Optimal', capsize=3, fillstyle='none')


# plt.legend(handler_map={f1: legend_handler.HandlerErrorbar(xerr_size=5)})

plt.xticks([1, 2, 3, 4, 5], ['100', '200', '300', '400', '500'],)


plt.tick_params(labelsize=12)
plt.grid(linestyle='--', linewidth=1)
# plt.xticks(range(1, LenSFC+1), range(1, LenSFC+1))

# plt.legend(numpoints=3, prop={'size': 16,'family':'Times New Roman'})

# plt.title(r'$\alpha_{max} = 3$', fontsize=16, fontname='Times New Roman')
ax1.set_xlabel(r'Number of flows', fontsize=18, fontname='Times New Roman')
ax1.set_ylabel('Total cost', fontsize=18, fontname='Times New Roman')

# ax1.legend(prop={'size': 12,'family':'Times New Roman'})

first_legend = plt.legend(handles=[fig2, fig3, fig5], loc='upper left', prop={'size': 16,'family':'Times New Roman'})

plt.gca().add_artist(first_legend)

plt.legend(handles=[fig4, fig1, fig6], loc='lower right', prop={'size': 16,'family':'Times New Roman'})


plt.savefig("/home/test/00-fast-journal/moop/fast-integration/fast-simulation/evaluation/results/3-ary-official/total_cost.pdf")

# ==================================================================================================================


fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)

# flierprops = dict(marker='o', markerfacecolor='green', markersize=12,linestyle='none')
# medianprops = dict(linestyle='-.', linewidth=1.5, color='firebrick')
fig1 = plt.errorbar(index, past_comm_cost_result, yerr=past_comm_cost_err, color=colors[5], linewidth=3, marker='x', markersize=15, label=r'Tabu', markeredgewidth=2, capsize=3, fillstyle='none')
fig2 = plt.errorbar(index, rast_comm_cost_result, yerr=rast_comm_cost_err, color=colors[4], linewidth=3, marker='^', markersize=8, label=r'Random', capsize=3, fillstyle='none')
fig3 = plt.errorbar(index, last_comm_cost_result, yerr=last_comm_cost_err, color=colors[3], linewidth=3, marker='o', markersize=8, label=r'LAST', capsize=3)
fig4 = plt.errorbar(index, base_comm_cost_result, yerr=base_comm_cost_err, color=colors[2], linestyle='--', linewidth=3, marker='.', markersize=5, label=r'No migration', capsize=3, fillstyle='none')
fig5 = plt.errorbar(index, bast_comm_cost_result, yerr=bast_comm_cost_err, color=colors[1], linewidth=3, marker='s', markersize=10, label=r'CAST', capsize=3, fillstyle='none')
fig6 = plt.errorbar(index, opt_comm_cost_result, yerr=opt_comm_cost_err, color=colors[0], linewidth=3, marker='.', label=r'Optimal', capsize=3, fillstyle='none')

plt.xticks([1, 2, 3, 4, 5], ['100', '200', '300', '400', '500'],)
# plt.rc('legend', fontsize=28)

plt.tick_params(labelsize=12)
plt.grid(linestyle='--', linewidth=1)
# plt.xticks(range(1, LenSFC+1), range(1, LenSFC+1))
# ax2.legend(ncol=2, labelspacing=0.05, prop={'size': 28,'family':'Times New Roman'})

# plt.legend(numpoints=3, prop={'size': 28,'family':'Times New Roman'})

# plt.title(r'$\alpha_{max} = 3$', fontsize=18, fontname='Times New Roman')
ax2.set_xlabel(r'Number of flows', fontsize=18, fontname='Times New Roman')
ax2.set_ylabel('Communication cost (Mbps)', fontsize=18, fontname='Times New Roman')

first_legend = plt.legend(handles=[fig4, fig2, fig1], loc='upper left', prop={'size': 16,'family':'Times New Roman'})

plt.gca().add_artist(first_legend)

plt.legend(handles=[fig6, fig3, fig5], loc='lower right', prop={'size': 16,'family':'Times New Roman'})

# plt.ylim([0, 1500])

plt.savefig("/home/test/00-fast-journal/moop/fast-integration/fast-simulation/evaluation/results/3-ary-official/comm_cost.pdf")

# ======================================================================================


fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)

# flierprops = dict(marker='o', markerfacecolor='green', markersize=12,linestyle='none')
# medianprops = dict(linestyle='-.', linewidth=1.5, color='firebrick')

# colors = ['tab:orange', 'green', 'firebrick', 'navy']
fig1 = plt.errorbar(index, past_buff_cost_result, yerr=past_buff_cost_err, color=colors[5], linewidth=3, marker='x', markersize=15, label=r'Tabu', markeredgewidth=2,capsize=3, fillstyle='none')
fig2 = plt.errorbar(index, rast_buff_cost_result, yerr=rast_buff_cost_err, color=colors[4], linewidth=3, marker='^', markersize=8, label=r'Random', capsize=3, fillstyle='none')
fig3 = plt.errorbar(index, last_buff_cost_result, yerr=last_buff_cost_err, color=colors[3], linewidth=3, marker='o', markersize=8, label=r'LAST', capsize=3)
fig4 = plt.errorbar(index, base_buff_cost_result, yerr=base_buff_cost_err, color=colors[2], linestyle='--', linewidth=2, marker='.', label=r'No migration', capsize=3, fillstyle='none')
fig5 = plt.errorbar(index, bast_buff_cost_result, yerr=bast_buff_cost_err, color=colors[1], linewidth=3, marker='s', markersize=10, label=r'CAST', capsize=3, fillstyle='none')
fig6 = plt.errorbar(index, opt_buff_cost_result, yerr=opt_buff_cost_err, color=colors[0], linewidth=3, marker='.', label=r'Optimal', capsize=3, fillstyle='none')

plt.xticks([1, 2, 3, 4, 5], ['100', '200', '300', '400', '500'],)


plt.tick_params(labelsize=12)
plt.grid(linestyle='--', linewidth=1)
# plt.xticks(range(1, LenSFC+1), range(1, LenSFC+1))

# plt.legend(numpoints=3, prop={'size': 16,'family':'Times New Roman'})

# plt.title(r'$\alpha_{max} = 3$', fontsize=16, fontname='Times New Roman')
ax3.set_xlabel(r'Number of flows', fontsize=18, fontname='Times New Roman')
ax3.set_ylabel(r'Buffering cost (Mb)', fontsize=18, fontname='Times New Roman')

first_legend = plt.legend(handles=[fig2, fig3, fig5], loc='upper left', prop={'size': 16,'family':'Times New Roman'})

plt.gca().add_artist(first_legend)

plt.legend(handles=[fig1, fig6, fig4], loc='center right', prop={'size': 16,'family':'Times New Roman'})


# ax3.legend(prop={'size': 12,'family':'Times New Roman'})

plt.savefig("/home/test/00-fast-journal/moop/fast-integration/fast-simulation/evaluation/results/3-ary-official/buff_cost.pdf")


# ==================================================================================

fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)

# flierprops = dict(marker='o', markerfacecolor='green', markersize=12,linestyle='none')
# medianprops = dict(linestyle='-.', linewidth=1.5, color='firebrick')
fig2 = plt.errorbar(index, rast_mig_time_result, yerr=rast_mig_time_err, color=colors[4], linewidth=3, marker='^', markersize=8, label=r'Random', capsize=3, fillstyle='none')
fig4 = plt.errorbar(index, bast_mig_time_result, yerr=bast_mig_time_err, color=colors[1], linewidth=3, marker='s', markersize=10, label=r'CAST', capsize=3, fillstyle='none')
fig3 = plt.errorbar(index, last_mig_time_result, yerr=last_mig_time_err, color=colors[3], linewidth=3, marker='o', markersize=8, label=r'LAST', capsize=3)
fig1 = plt.errorbar(index, past_mig_time_result, yerr=past_mig_time_err, color=colors[5], linewidth=3, marker='x', markersize=15, label=r'Tabu', markeredgewidth=2, capsize=3, fillstyle='none')
# plt.errorbar(index, base_mig_time_result, yerr=base_mig_time_err, color=colors[2], linewidth=2, marker='s', label=r'No migration', capsize=5, fillstyle='none')
fig5 = plt.errorbar(index, opt_mig_time_result, yerr=opt_mig_time_err, color=colors[0], linewidth=3, marker='.', label=r'Optimal', capsize=3, fillstyle='none')


plt.xticks([1, 2, 3, 4, 5], ['100', '200', '300', '400', '500'],)


plt.tick_params(labelsize=12)
plt.grid(linestyle='--', linewidth=1)
# plt.xticks(range(1, LenSFC+1), range(1, LenSFC+1))

# plt.legend(numpoints=3, prop={'size': 16,'family':'Times New Roman'})

# plt.title(r'$\alpha_{max} = 3$', fontsize=16, fontname='Times New Roman')
ax4.set_xlabel(r'Number of flows', fontsize=18, fontname='Times New Roman')
ax4.set_ylabel(r'Total transfer time (ms)', fontsize=18, fontname='Times New Roman')

# ax4.legend(prop={'size': 12,'family':'Times New Roman'})

ax4.legend(ncol=2, labelspacing=0.05, prop={'size': 16,'family':'Times New Roman'}, loc='center')

plt.savefig("/home/test/00-fast-journal/moop/fast-integration/fast-simulation/evaluation/results/3-ary-official/mig_time.pdf")

# ================================================================================

fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111)

# flierprops = dict(marker='o', markerfacecolor='green', markersize=12,linestyle='none')
# medianprops = dict(linestyle='-.', linewidth=1.5, color='firebrick')
plt.errorbar(index, past_exec_time_result, yerr=past_exec_time_err, color=colors[5], linewidth=2, marker='x', markersize=15, label=r'Tabu', markeredgewidth=2, capsize=3, fillstyle='none')
plt.errorbar(index, rast_exec_time_result, yerr=rast_exec_time_err, color=colors[4], linewidth=2, marker='^', markersize=8, label=r'Random', capsize=3, fillstyle='none')
plt.errorbar(index, last_exec_time_result, yerr=last_exec_time_err, color=colors[3], linewidth=2, marker='o', markersize=8, label=r'LAST', capsize=3)
plt.errorbar(index, base_exec_time_result, yerr=base_exec_time_err, color=colors[2], linewidth=2, linestyle='--', marker='.', markersize=5, label=r'No migration', capsize=3, fillstyle='none')
plt.errorbar(index, bast_exec_time_result, yerr=bast_exec_time_err, color=colors[1], linewidth=2, marker='s', markersize=10, label=r'CAST', capsize=3, fillstyle='none')
plt.errorbar(index, opt_exec_time_result, yerr=opt_exec_time_err, color=colors[0], linewidth=2, marker='.', label=r'Optimal', capsize=3, fillstyle='none')


plt.xticks([1, 2, 3, 4, 5], ['100', '200', '300', '400', '500'],)


plt.tick_params(labelsize=12)
plt.grid(linestyle='--', linewidth=1)
# plt.xticks(range(1, LenSFC+1), range(1, LenSFC+1))

# plt.legend(numpoints=3, prop={'size': 16,'family':'Times New Roman'})

# plt.title(r'$\alpha_{max} = 3$', fontsize=16, fontname='Times New Roman')
ax5.set_xlabel(r'Number of flows', fontsize=16, fontname='Times New Roman')
ax5.set_ylabel('Execution time', fontsize=16, fontname='Times New Roman')

ax5.legend(prop={'size': 12,'family':'Times New Roman'})

plt.savefig("/home/test/00-fast-journal/moop/fast-integration/fast-simulation/evaluation/results/3-ary-official/exec_time.pdf")

# ===============================================================================

fig6 = plt.figure(6)
ax6 = fig6.add_subplot(111)

# flierprops = dict(marker='o', markerfacecolor='green', markersize=12,linestyle='none')
# medianprops = dict(linestyle='-.', linewidth=1.5, color='firebrick')
plt.errorbar(index, rast_comp_cost_result, yerr=rast_comp_cost_err, color=colors[4], linewidth=3, marker='^', markersize=8, label=r'Random', capsize=3, fillstyle='none')
plt.errorbar(index, bast_comp_cost_result, yerr=bast_comp_cost_err, color=colors[1], linewidth=3, marker='s', markersize=10, label=r'CAST', capsize=3, fillstyle='none')
plt.errorbar(index, last_comp_cost_result, yerr=last_comp_cost_err, color=colors[3], linewidth=3, marker='o', markersize=8, label=r'LAST', capsize=3)
plt.errorbar(index, past_comp_cost_result, yerr=past_comp_cost_err, color=colors[5], linewidth=3, marker='x', markersize=15, label=r'Tabu', markeredgewidth=2, capsize=3, fillstyle='none')
plt.errorbar(index, opt_comp_cost_result, yerr=opt_comp_cost_err, color=colors[0], linewidth=3, marker='.', label=r'Optimal', capsize=3, fillstyle='none')


plt.xticks([1, 2, 3, 4, 5], ['100', '200', '300', '400', '500'],)


plt.tick_params(labelsize=12)
plt.grid(linestyle='--', linewidth=1)
# plt.xticks(range(1, LenSFC+1), range(1, LenSFC+1))

# plt.legend(numpoints=3, prop={'size': 16,'family':'Times New Roman'})

# plt.title(r'$\alpha_{max} = 3$', fontsize=16, fontname='Times New Roman')
ax6.set_xlabel(r'Number of flows', fontsize=18, fontname='Times New Roman')
ax6.set_ylabel('Compuration cost', fontsize=18, fontname='Times New Roman')

#ax6.legend(prop={'size': 12,'family':'Times New Roman'})

ax6.legend(ncol=2, labelspacing=0.05, prop={'size': 16,'family':'Times New Roman'}, loc='center')

plt.savefig("/home/test/00-fast-journal/moop/fast-integration/fast-simulation/evaluation/results/3-ary-official/comp_cost.pdf")



plt.show()
