import numpy as np
num_keys=4

BODY_PARTS_KPT_IDS = [[0,1],[0, 2], [0, 3]]

BODY_PARTS_PAF_IDS = ([0, 1], [2, 3],[4,5])


kpt_names = ['head', 'left_engine','right_engine','bottom_engine']

sigmas = np.array([0.5, 0.5, 0.5,0.5],dtype=np.float32) / 10.0

#BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
#                      [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [0, 16], [0, 17]]
#BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
#                      [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27])

min_score=0.2
#kpt_names = ['nose', 'neck',
#                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
#                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
#                 'r_eye', 'l_eye',
#                 'r_ear', 'l_ear']
#sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
#                      dtype=np.float32) / 10.0

save=True
print_pose_entries=True
print_allkeypoints=True
