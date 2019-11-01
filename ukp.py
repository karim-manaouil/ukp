import sys
import os
import subprocess
from timeit import default_timer as timer
from datetime import timedelta
from random import randrange

# Some dirty hack:
global_fhandle = 0

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
    current_obj_i = 0
    current_obj = twins[current_obj_i].obj
    cont = True

    ukp_sol_o = ukp_solution()

    while current_capacity > 0 and cont:
        if (ukp_obj.w[current_obj] > current_capacity):
            current_obj_i += 1
            if current_obj_i < len(ukp_obj.p):
                current_obj = twins[current_obj_i].obj
            else:
                cont = False
            continue

        ukp_select_object(ukp_sol_o, current_obj)
        ukp_sol_o.total += ukp_obj.p[current_obj]
        ukp_sol_o.tw += ukp_obj.w[current_obj]

        current_capacity -= ukp_obj.w[current_obj]

    return ukp_sol_o

# total value heuristic
def ukp_tv(ukp_object):
    ukp_object.validate()
    cc = ukp_object.capacity # left capacity in each iteration

    rem_objects = list(range(0, len(ukp_object.p))) # Remaining objects
    ukp_sol_o = ukp_solution()

    while (cc > 0 and len(rem_objects) > 0):
        max_metric = 0; selected = -1
        for object in rem_objects:
            metric = ukp_object.p[object] * int(cc/ukp_object.w[object])
            if metric > max_metric:
                max_metric = metric
                selected = object

        cc = cc - ukp_object.w[selected]
        if cc < 0:
            break

        ukp_select_object(ukp_sol_o, selected)
        ukp_sol_o.total += ukp_object.p[selected]
        ukp_sol_o.tw += ukp_object.w[selected]
        rem_objects.remove(selected)

    return ukp_sol_o

# Weight-Ordered heuristic solution
def ukp_wo(ukp_object):
    ukp_object.validate()
    cc = ukp_object.capacity # left capacity in each iteration
    temp = list(ukp_object.w)
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
            ukp_sol_o.tw += ukp_object.w[index[i]]
            cc -= ukp_object.w[index[i]]
        else:
            i = i+1
    return ukp_sol_o

def ukp_ga(ukp_obj, ukp_obj_bin, generations, mutation_percentage):

    OIndex = create_ordered_index(ukp_obj_bin)

    s1 = generate_random_population (ukp_obj_bin)
    s2 = generate_random_population (ukp_obj_bin)

    generation = 0
    while generation < generations :

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

        generation += 1

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
    i = 0; m = 0; k = -1

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


import math
import random

ALPHA = 0.85


def ukp_binarize_simulated_annealing(ukp_obj):
    weight_cost = ([], [])
    temp = ukp_binarize(ukp_obj)

    weight_cost = [(temp.w[i], temp.p[i]) for i in range(len(temp.w))]
    return weight_cost


def ukp_annealing_algorithm(number, capacity, weight_cost, init_temp=100, steps=100):
    start_sol = init_solution(weight_cost, capacity)
    best_cost, solution = simulate(start_sol, weight_cost, capacity, init_temp, steps)
    best_combination = [0] * number
    ukp_sol_o = ukp_solution()
    for idx in solution:
        best_combination[idx] = 1
    return best_combination


def init_solution(weight_cost, max_weight):
    """Used for initial solution generation.
    By adding a random item while weight is less max_weight
    """
    solution = []
    allowed_positions = list(range(len(weight_cost)))
    while len(allowed_positions) > 0:
        idx = random.randint(0, len(allowed_positions) - 1)
        selected_position = allowed_positions.pop(idx)
        if get_cost_and_weight_of_knapsack(solution + [selected_position], weight_cost)[1] <= max_weight:
            solution.append(selected_position)
        else:
            break
    return solution


def get_cost_and_weight_of_knapsack(solution, weight_cost):
    """Get cost and weight of knapsack - fitness function
    """
    cost, weight = 0, 0
    for item in solution:
        weight += weight_cost[item][0]
        cost += weight_cost[item][1]
    return cost, weight


def moveto(solution, weight_cost, max_weight):
    """All possible moves are generated"""
    moves = []
    for idx, _ in enumerate(weight_cost):
        if idx not in solution:
            move = solution[:]
            move.append(idx)
            if get_cost_and_weight_of_knapsack(move, weight_cost)[1] <= max_weight:
                moves.append(move)
    for idx, _ in enumerate(solution):
        move = solution[:]
        del move[idx]
        if move not in moves:
            moves.append(move)
    return moves


def simulate(solution, weight_cost, max_weight, init_temp, steps):
    """Simulated annealing approach for Knapsack problem"""
    temperature = init_temp

    best = solution
    best_cost = get_cost_and_weight_of_knapsack(solution, weight_cost)[0]

    current_sol = solution
    while True:
        current_cost = get_cost_and_weight_of_knapsack(best, weight_cost)[0]
        for i in range(0, steps):
            moves = moveto(current_sol, weight_cost, max_weight)
            idx = random.randint(0, len(moves) - 1)
            random_move = moves[idx]
            delta = get_cost_and_weight_of_knapsack(random_move, weight_cost)[0] - best_cost
            if delta > 0:
                best = random_move
                best_cost = get_cost_and_weight_of_knapsack(best, weight_cost)[0]
                current_sol = random_move
            else:
                if math.exp(delta / float(temperature)) > random.random():
                    current_sol = random_move

        temperature *= ALPHA
        if current_cost >= best_cost or temperature <= 0:
            break
    return best_cost, best


def read_benchmark_instance(path):
    ukp_obj = ukp(0, [], [])

    try :
        f = open(path, "r")
    except IOError:
        return -1

    state = 0 # Data read phase not yet

    for line in f:
        s = line.split()
        if len(s) > 1:
            if s[0] == "n:":
                n = int(s[1])
            elif s[0] == "c:":
                ukp_obj.capacity = int(s[1])
            elif s[0] == "begin" :
                state = 1 # Begin data read phase
            elif s[0] == "end" :
                break # End data read phase
            elif state == 1:
                ukp_obj.w.append(int(s[0]))
                ukp_obj.p.append(int(s[1]))

    return ukp_obj

# (type, ukp_obj, ...)
def execute_instance(*k):
    type = k[0]
    ukp_o = k[1]
    start = timer()

    if type == "ukp_dno":
        solution = ukp_dno(ukp_o)

    elif type == "ukp_tv":
        solution = ukp_tv(ukp_o)

    elif type == "ukp_wo":
       solution = ukp_wo(ukp_o)

    # (type, ukp_obj, generations, mutation)
    elif "ukp_ga" in type:
        ukp_o_bin = ukp_binarize(ukp_o)
        selected = ukp_ga(ukp_o, ukp_o_bin, k[2], k[3])
        solution = ukp_debinarize_solution(ukp_o_bin, selected)
    elif type == "ukp_sa":
        ukp_bin = ukp_binarize(ukp_o)
        weight_coast_vector = ukp_binarize_simulated_annealing(ukp_o)
        selected = ukp_annealing_algorithm(len(weight_coast_vector), ukp_o.capacity, weight_coast_vector,\
                                           SA_PARAMETERS.TEMPERATURE,SA_PARAMETERS.NBRE_CYCLES)
        solution = ukp_debinarize_solution(ukp_bin, selected)

    else:

        return

    time = str(timedelta(seconds=timer() - start))

    global global_fhandle

    if global_fhandle != 0:
        global_fhandle.write(type + ',' + time + ',' + str(solution.total) + ',' + str(solution.tw) + '\n')

    print (type + ',' + time + ',' + str(solution.total) + ',' + str(solution.tw))

    return {"sol":solution, "time":time}

def do_benchmark(file, fhandle):
    ukp_obj = read_benchmark_instance(file)

    global  global_fhandle
    global_fhandle = fhandle

    # fhandle.write("Benchmarking " + os.path.basename(file) + \
    #       ": N=" + str(len(ukp_obj.p)) + " C=" + str(ukp_obj.capacity) + "\n")

    fhandle.write("type,time,value,weight\n")

    execute_instance("ukp_wo", ukp_obj)
    execute_instance("ukp_dno", ukp_obj)
    execute_instance("ukp_tv", ukp_obj)
    execute_instance("ukp_sa", ukp_obj)
    execute_instance("ukp_ga.10.20", ukp_obj, 10, 0.20)
    execute_instance("ukp_ga.100.20", ukp_obj, 100, 0.20)
    execute_instance("ukp_ga.1000.20", ukp_obj, 1000, 0.20)
    execute_instance("ukp_ga.100.60", ukp_obj, 100, 0.60)
    execute_instance("ukp_ga.1000.60", ukp_obj, 1000, 0.60)
    execute_instance("ukp_ga.1000.10", ukp_obj, 1000, 0.10)


def run_bechmarks(path, tmp):
    for r, d, f in os.walk(path):
        for file in f:
            fhandle = open(os.path.join(tmp, file), "w+")
            do_benchmark(os.path.join(r, file), fhandle)
            fhandle.close()

########################### Parameter definition ##########################

class GA_PARAMETERS:
    GENERATIONS = 1000
    MUTATION_RATIO = 0.10


class SA_PARAMETERS:
    TEMPERATURE = 100
    NBRE_CYCLES = 100

def main():
    assets = [
        {"desc":"Generated assets", "path":"assets/upk/generated/"},
        {"desc":"EDUK2000", "path":"assets/upk/EDUK2000"}
    ]

    folder = "ukp_run"

    try :
        subprocess.run(["rm","-rf",folder])
        os.mkdir(folder)
    except :
        return -1

    for asset in assets:
        run_bechmarks(asset["path"], folder)

main()

# # (c, p, w)
    # instance = ukp(10, [5, 6, 7, 8], [1, 2, 1, 3])
