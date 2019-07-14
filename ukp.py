import sys
import time
from random import randrange

#class objet
class objet:
    profit = 0
    weight = 0
    def __init__(self, profit, weight):
        self.weight = weight
        self.profit = profit

# This is the class that contains the information
# of an instance : capacity, weights list and profits list

class ukp:
    capacity = 0
    p = []  # Profits Array
    w = []  # Weights Array
    rep = [] # [C/Wi] to binarize ukp

    def __init__(self, capacity, p, w):
        self.capacity = capacity
        self.p = list(p)
        self.w = list(w)

    def validate(self):
        if self.capacity < 0 or len(self.p) != len(self.w):
            print
            "Validation error"
            sys.exit()

# This is useful when you need a pair of
# (value, object) in your algorithm
class rowtwin:
    def __init__(self, obj, row):
        self.row = row
        self.obj = obj

# This is what all the algorithms must return
# the total profit, the list of taken objects
# and the list that contains how many times
# each object has been taken
class ukp_solution:
    def __init__(self):
        self.total = 0
        self.tw = 0  # Total weight
        self.taken = []
        self.ttimes = []

# This method inserts an object into the solutions structure
def ukp_select_object (ukp_solution_o, object):
    ukp_solution_o.taken.append(object)


############################# Density Oredered Heuristic #############################

# Density ordered heuristic, it takes a ukp object as parameter
# and return a ukp_solution
def ukp_dno(ukp_obj):
    twins = []
    ukp_obj.validate()

    for i in range(0, len(ukp_obj.p)):
        tmp_row = int(ukp_obj.p[i]/ukp_obj.w[i])
        twins.append(rowtwin(i, tmp_row))

    twins.sort(key=lambda x: x.row, reverse=True)
    # for twin in twins:
    #     print (str(twin.obj) + ":" + str(twin.row))

    current_capacity = ukp_obj.capacity
    total_profit = 0
    current_obj_i = 0
    current_obj = twins[current_obj_i].obj
    cont = True

    ukp_sol_o = ukp_solution()

    while current_capacity > 0 and cont:
        if (ukp_obj.w[current_obj] > current_capacity):
            if ++current_obj_i < len(ukp_obj.p):
                current_obj = twins[current_obj_i].obj
            else:
                cont = False
            continue

        ukp_select_object(ukp_sol_o, current_obj)
        ukp_sol_o.total += ukp_obj.p[current_obj]
        current_capacity -= ukp_obj.w[current_obj]

    return ukp_sol_o

############################# End of Density Oredered Heuristic #############################


############################# Total Oredered Heuristic #############################

# total value heuristic
def ukp_tv(ukp_object):
    ukp_object.validate()
    cc = ukp_object.capacity # left capacity in each iteration

    rem_objects = list(range(0, len(ukp_object.p))) # Remaining objects
    ukp_sol_o = ukp_solution()

    while (cc > 0 and len(rem_objects) > 0):
        max_metric = 0; selected = -1
        for object in rem_objects:
            print(rem_objects)
            metric = ukp_object.p[object] * int(cc/ukp_object.w[object])
            if metric > max_metric:
                max_metric = metric
                selected = object

        cc = cc - ukp_object.w[selected]
        if cc < 0:
            break

        ukp_select_object(ukp_sol_o, selected)
        ukp_sol_o.total += ukp_object.p[selected]
        rem_objects.remove(selected)

    return ukp_sol_o


############################# End of Total Oredered Heuristic #############################


############################# Weight Oredered Heuristic #############################

# Weight-Ordered heuristic solution
def ukp_wo(ukp_object):
    ukp_object.validate()
    cc = ukp_object.capacity # left capacity in each iteration
    temp = ukp_object.w.copy()
    index = list(range(0, len(ukp_object.p)))  # index des objets trie
    #ordonner l index du tableau
    ukp_sol_o = ukp_solution()
    #organiser l index
    for iter_num in range(len(ukp_object.w) - 1, 0, -1):
        for idx in range(iter_num):
            if temp[idx] > temp[idx + 1]:
                temp[idx], temp[idx + 1] = temp[idx+1], temp[idx]
                index[idx], index[idx + 1] = index[idx+1], index[idx]
    i = 0
    while cc > 0 and i < len(ukp_object.w):
        if ukp_object.w[index[i]] < cc:
            ukp_select_object(ukp_sol_o, index[i])
            ukp_sol_o.total += ukp_object.p[index[i]]
            cc -= ukp_object.w[index[i]]
        else:
            i = i+1
    return ukp_sol_o

############################# End of Weight Oredered Heuristic #############################


############################# Genetic Algorithm #############################

def ukp_ga(ukp_obj, times, mutation_percentage):

    ukp_obj_bin = ukp_binarize(ukp_obj)

    OIndex = create_ordered_index(ukp_obj_bin)

    s1 = generate_random_population (ukp_obj_bin)
    s2 = generate_random_population (ukp_obj_bin)

    time = 0
    while time < times :

        fs1 = get_fitness_of(ukp_obj_bin, s1)
        fs2 = get_fitness_of(ukp_obj_bin, s2)

        best = s1 if fs1 > fs2 else s2 # Best individual for this iteration

        child = cross_over (s1, s2)
        child = mutate (child, mutation_percentage)

        repair (ukp_obj_bin, child, OIndex)

        # replace worst individual with child
        if fs1 > fs2 :
            s2 = child
        else :
            s1 = child

        fsc = get_fitness_of(ukp_obj_bin, child)

        # Replace best indivual with child if f(child) > f(best)
        if fs1 > fs2 and fsc > fs1:
            s1 = child
        elif fs2 > fs1 and fsc > fs2:
            s2 = child

        time += 1

    return best

# This returns the list of object indices ordered by Pi/Wi
def create_ordered_index(bukp_obj):
    index_list = []
    pw_list = []

    # 10, [5, 6, 7, 8], [1, 2, 1, 3]
    for i in range(0, len(bukp_obj.w)):
        pw_list.append(int(bukp_obj.p[i]/bukp_obj.w[i]))

    i = 0; n = len(pw_list)
    while (i < n):
        next = -1; nextIndex = -1
        for j in range(0, n):
            if pw_list[j] != -1 and pw_list[j] > next:
                next = pw_list[j]
                nextIndex = j

        rep = int(bukp_obj.capacity/bukp_obj.w[nextIndex])
        for k in range(0, rep):
            index_list.append(nextIndex + k)
            pw_list[nextIndex + k] = -1

        i += rep

    return index_list

# Calculates solution vector total weight
def get_sv_weight(bukp_obj, sv):
    i = 0; m = 0; k = -1;
    weight = 0

    while (i < len(bukp_obj.w)):
        k += 1
        m += bukp_obj.rep[k]
        while (i < m):
            weight += bukp_obj.w[m - bukp_obj.rep[k]] * sv[i]
            i += 1

    return weight

def get_fitness_of(bukp_obj, sv):
    i = 0; m = 0; k = -1;
    fitness = 0

    while (i < len(bukp_obj.w)):
        k += 1
        m += bukp_obj.rep[k]
        while (i < m):
            fitness += bukp_obj.p[m - bukp_obj.rep[k]] * sv[i]
            i += 1

    return fitness

def generate_random_population(ukp_obj):
    l = len(ukp_obj.w)
    vect = [0] * l
    available = []

    for i in range(0, l):
        available.append(i)

    R = 0
    robj = randrange(len (available))
    available.pop(robj)

    while R + ukp_obj.w[robj] < ukp_obj.capacity:
        vect[robj] = 1
        R += ukp_obj.w[robj]
        robj = randrange(len(available))
        available.pop(robj)

    return vect

# A new child is born here
def cross_over(father, mother):
    child  = [0] * len(father)

    for i in range(0, len(father)):
        if randrange(0, 2) == 0 :
            child[i] = father[i]
        else :
            child[i] = mother[i]

    return child

def mutate(child, percentage):
    # Mutate 20% of the genome
    mutations = int(len(child)*percentage)

    for i in range(0, mutations):
        chromosome = randrange(0, len(child))
        child[chromosome] = 1 - child[chromosome] # 0 becomes 1 and 1 becomes 0

    return child

# Genetic mutation may result in deformation so
# the individual must be repaired
def repair(bukp_obj, child, OIndex):
    R = get_sv_weight(bukp_obj, child)

    # Nothing to repair
    if R <= bukp_obj.capacity:
        return child

    # Drop phase
    for i in range (0, len(OIndex)):
        j = OIndex[len(OIndex) - i - 1]
        if child[j] == 1 :
            if R > bukp_obj.capacity :
                child[j] = 0
                R -= bukp_obj.w[j]

    # Add phase
    for i in range(0, len(OIndex)):
        j = OIndex[i]
        if child[j] == 0:
            if R + bukp_obj.w[j] < bukp_obj.capacity:
                child[j] = 1
                R += bukp_obj.w[j]

    return child

def ukp_binarize(ukp_obj):
    bin_obj = ukp(0, [], [])
    bin_obj.capacity = ukp_obj.capacity

    for i in range(0, len(ukp_obj.w)):
        hmt = int(ukp_obj.capacity/ukp_obj.w[i]) # How many times ?
        bin_obj.rep.insert(i, hmt)
        for j in range(0, hmt): # Insert element i #hmt times
            bin_obj.w.append(ukp_obj.w[i])
            bin_obj.p.append(ukp_obj.p[i])

    return bin_obj

# This us actually a O(n) algorithm, it just looks too complex
def ukp_debinarize_solution (ukp_obj, sv):
    ukp_sol_o = ukp_solution()
    i = 0; m = 0; k = -1;

    while (i < len(ukp_obj.w)):
        k += 1
        m += ukp_obj.rep[k]
        while (i < m):
            if sv[i] == 1:
                ukp_select_object (ukp_sol_o, k)
                ukp_sol_o.total += ukp_obj.p[m - ukp_obj.rep[k]]
                ukp_sol_o.tw += ukp_obj.w[m - ukp_obj.rep[k]]
            i += 1

    return ukp_sol_o

############################# End of Genetic Algorithm #############################

def execute_instance(type, ukp_o):
    start = time.localtime()

    if type == "ukp_dno":
        solution = ukp_dno(ukp_o)
    elif type == "ukp_tv":
        solution = ukp_tv(ukp_o)
    elif type =="ukp_wo":
       solution = ukp_wo(ukp_o)
    else:
        return

    end = time.localtime()

    print (type)
    print ("Executed in " + str(end.tm_sec - start.tm_sec) + " secs")
    print ("Total profit = " + str(solution.total))
    print ("chosen objects:times")
    for i in range(0, len(solution.taken)):
        print(str(solution.taken[i]) + ":" + str(solution.ttimes[i]))


def main():
    # (c, p, w)
    instance = ukp(10, [5, 6, 7, 8], [1, 2, 1, 3])

    #execute_instance("ukp_dno", instance)
    #execute_instance("ukp_wo", instance)

    best = ukp_ga(instance, 10, 0.20)
    sol = ukp_debinarize_solution(ukp_binarize(instance), best)

    print ("ok")

main()