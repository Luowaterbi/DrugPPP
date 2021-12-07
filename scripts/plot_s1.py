import numpy as np
import re
import matplotlib.pyplot as plt

def grab_one_file(fname):
    pat = re.compile(r"[\d\.]+")
    all_stats = []
    with open(fname, "r", encoding="utf8") as ff:
        for line in ff:
            segs = re.findall(r"[\d\.]+", line.strip())
            if len(segs) == 4:
                all_stats.append([float(x) for x in segs])
               
    stats = np.array(all_stats)
    return stats
    
    
stats_ours = grab_one_file("saign.bs_64.ep_200.patience_10.log")
stats_baseline = grab_one_file("cigin.bs_32.ep_100.lr_0.001.inter_dot.res_all.ro_set2set.DATA.MNSol.sd_0.log")

plt.plot(stats_baseline[:,0], stats_baseline[:, 1], 'r-', label="(theirs) training loss")
plt.plot(stats_baseline[:,0], stats_baseline[:, 2], 'g-', label="(theirs) dev loss")


plt.plot(stats_ours[:,0], stats_ours[:, 1], 'b-', label="(ours) training loss")
plt.plot(stats_ours[:,0], stats_ours[:, 2], 'c-', label="(ours) dev loss")


plt.legend()
plt.grid(True)
plt.show()   
            
       
    


