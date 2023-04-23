#state formulation S is a dictionary of (v and w)
# where v = v1 , v2 , ... , vn
# and w = w1 , w2 , ... , wn

# node state data vk for nodes. 
# index 
# 1 : 1 if pin-supported, else 0 
# 2 Load intensity [kN] at the node in x direction (load case 1)
# 3 Load intensity [kN] at the node in y direction (load case 1)
# .
# .
# .
# 2nL Load intensity [kN] at the node in x direction (load case nL)
# 2nL+1 Load intensity [kN] at the node in y direction (load case nL)

# member state data wi for members
# index
# 1 cos α : α: the angle of the member with respect to positive x direction
# 2 sin α : α: the angle of the member with respect to positive x direction
# 3 Member length
# 1 if remained, 0 if removed
# Stress safety ratio (load case 1) 
# .
# .
# .
# nL+4 Stress safety ratio (load case nL)


