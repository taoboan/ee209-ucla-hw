import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
import sys
from scipy.spatial import ConvexHull

length = 80.0
width = 85.0
r = 20.0
K_distance = 0.2
K_angle1 = 1.2
K_angle2 = -0.2
dt = 0.1


class Node():

    def __init__(self, state):
        self.state = state
        self.parent = None

# Problem 2(a)
def near_node(node_list, target):

    min_distance = sys.maxsize
    min_index = sys.maxsize
    for i, node in enumerate(node_list):
        distance = np.linalg.norm(target - node.state)
        if distance < min_distance:
            NearNode = node
            min_distance = distance
            min_index = i
    return NearNode, min_index


# Problem 2(b)

def trajectory(initial, target, type = 0):

    x0 = initial[0]
    y0 = initial[1]
    angle0 = initial[2]
    x_end = target[0]
    y_end = target[1]
    angle_end = target[2]

    current_x = x0
    current_y = y0
    current_angle = angle0
    delta_x = x_end - current_x
    delta_y = y_end - current_y
    traj = []
    input_traj = []
    distance_remain = np.sqrt(delta_x ** 2 + delta_y ** 2)
    t = 0

    while t < 1.0 and distance_remain > 1:

        traj.append(np.array([current_x, current_y, current_angle]))
        delta_x = x_end - current_x
        delta_y = y_end - current_y
        distance_remain = np.sqrt(delta_x ** 2 + delta_y ** 2)
        goal_heading_angle = (np.arctan2(delta_y, delta_x) - current_angle + np.pi) % (2 * np.pi) - np.pi
        goal_change = (angle_end - current_angle - goal_heading_angle + np.pi) % (2 * np.pi) - np.pi

        v = K_distance * distance_remain

        w = K_angle1 * goal_heading_angle + K_angle2 * goal_change

        w_right = (2 * v + w * width) / (2 * r)

        w_left = (2 * v - w * width) / (2 * r)

        input_traj.append(np.array([w_left, w_right]))

        if goal_heading_angle > np.pi / 2 or goal_heading_angle < -np.pi / 2:
            v = -v

        current_angle = current_angle + w * dt
        current_x = current_x + v * np.cos(current_angle) * dt
        current_y = current_y + v * np.sin(current_angle) * dt
        if type is 0:
            t += dt

    return traj, input_traj

# Problem 2(c)

def Ospace(initial, target, obstacle_list):


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim((0, 1500))
    plt.ylim((0, 1500))
    plt.grid()

    circle0 = plt.Circle((initial[0], initial[1]), 20, color='red')
    circle1 = plt.Circle((target[0], target[1]), 20, color='blue')
    ax.add_patch(circle0)
    ax.add_patch(circle1)

    for item in obstacle_list:
        obstacle = plt.Rectangle(item[0:2], item[2], item[3], color='green')
        ax.add_patch(obstacle)
 #   plt.show()
    return ax


def Cspace(initial, target, obstacle_list):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    plt.xlim((0, 1500))
    plt.ylim((0, 1500))
    C_obstacle = {}
    for item in obstacle_list:
        angle_possibility = np.linspace(-np.pi, np.pi, 100)
        points = []
        for angle in angle_possibility:
            cordinates = [[item[0], item[1]], [item[0] + item[2], item[1]],
                 [item[0], item[1] + item[3]], [item[0] + item[2], item[1] + item[3]]]
            for x in cordinates:
                point1 = [x[0] + 70 * cos(angle) + 42.5 * cos(angle - np.pi / 2),x[1] + 70 * sin(angle) + 42.5 * sin(angle - np.pi / 2)]
                point2 = [x[0] + 70 * cos(angle) + 42.5 * cos(angle + np.pi / 2),x[1] + 70 * sin(angle) + 42.5 * sin(angle + np.pi / 2)]
                point3 = [x[0] + 10 * cos(angle + np.pi) + 42.5 * cos(angle - np.pi / 2),x[1] + 10 * sin(angle + np.pi) + 42.5 * sin(angle - np.pi / 2)]
                point4 = [x[0] + 10 * cos(angle + np.pi) + 42.5 * cos(angle + np.pi / 2),x[1] + 10 * sin(angle + np.pi) + 42.5 * sin(angle + np.pi / 2)]
                points.append(point1)
                points.append(point2)
                points.append(point3)
                points.append(point4)
        points = np.array(points)
        obstacle_range = ConvexHull(points)
        C_obstacle[item] = obstacle_range
        plt.plot(points[obstacle_range.vertices, 0],  points[obstacle_range.vertices,1], 'r--', lw=2)
    circle0 = plt.Circle((initial[0], initial[1]), 20, color='red')
    circle1 = plt.Circle((target[0], target[1]), 20, color='blue')
    ax.add_patch(circle0)
    ax.add_patch(circle1)
    plt.grid()
    plt.show()
    return C_obstacle

# Problem 2(d)
def collision_check(trajactory, C_obstacle):

    collision = False
    for state in trajactory:

        for item in C_obstacle.keys():
            old_hull = C_obstacle[item]
            #print("1.",old_hull.points.shape,"2.",np.array(([state[0:2]])).shape)
            new_hull = ConvexHull(np.concatenate((old_hull.points, [state[0:2]])))
            if np.array_equal(new_hull.vertices, old_hull.vertices):
                collision = True
    return collision


# Problem 2(e)

def rrt_planner(initial, target, random_boundary, Cspace_env, try_limit=1800):

    start_node = Node(initial)
    nodelist = [start_node]
    try_times = 0
    np.random.seed(1)

    while try_times <= try_limit:

        distance_goal = np.linalg.norm(target - nodelist[-1].state)
        if distance_goal < 50:
            break

        rrt_random = np.zeros(3)
        for i in range(3):
            rrt_random[i] = np.random.uniform(random_boundary[i][0], random_boundary[i][1])
        try_times += 1

        NearNode, near_index = near_node(nodelist, rrt_random)

        current_traj, _ = trajectory(NearNode.state.copy(), rrt_random, type=0)

        if not collision_check(current_traj, Cspace_env):

            newNode = Node(current_traj[-1])

            newNode.parent = near_index

            nodelist.append(newNode)

    path = [target]
    end_index = len(nodelist) - 1
    while nodelist[end_index].parent:
        node = nodelist[end_index]
        path.append(node.state)
        end_index = node.parent
    path.append(initial)

    print("rrt planner length:", len(path))

    traj_out = []
    i=0
    while i <= len(path) -2:
        traj_part, _ = trajectory(path[i], path[i + 1], type = 1)
        traj_out = traj_out + traj_part
        i = i + 1
    return traj_out

def draw_car(trajactory_points,obstacle):
    initial_position = trajactory_points[0]
    last_position = trajactory_points[-1]
    ax = Ospace(initial_position, last_position, obstacle)
    for states in trajactory_points:
        angle = states[2]
        robot = plt.Rectangle([states[0]+70*cos(angle+np.pi)+42.5*cos(angle-np.pi/2),states[1]+70*sin(angle+np.pi)+42.5*sin(angle-np.pi/2)], 80, 85, np.degrees(angle), facecolor='white', edgecolor='yellow')
        ax.add_patch(robot)

    plt.show()
    plt.ioff()

def rrt_greedy(initial, target, random_boundary, Cspace_env, try_limit=1800):

    start_node = Node(initial)
    nodelist = [start_node]
    try_times = 0
    np.random.seed(1)

    while try_times <= try_limit:

        distance_goal = np.linalg.norm(target - nodelist[-1].state)
        if distance_goal < 50:
            break

        rrt_random = np.zeros(3)
        for i in range(3):
            rrt_random[i] = np.random.uniform(random_boundary[i][0], random_boundary[i][1])
        try_times += 1

        NearNode, near_index = near_node(nodelist, rrt_random)

        current_traj, _ = trajectory(NearNode.state.copy(), rrt_random, type=0)

        if not collision_check(current_traj, Cspace_env):

            newNode = Node(current_traj[-1])

            newNode.parent = near_index

            nodelist.append(newNode)

    path = [target]

    end_index = len(nodelist) - 1

    while nodelist[end_index].parent is not None:

        node = nodelist[end_index]
        last_node = nodelist[node.parent]
        temp_traj, _ = trajectory(node.state, last_node.state, type=1)
        while not collision_check(temp_traj, Cspace_env):
            index = last_node.parent
            temp = last_node
            last_node = nodelist[index]
            temp_traj, _ = trajectory(node.state, last_node.state, type=1)
        path.append(temp.state)
        end_index = temp.parent

    path.append(initial)
    print("rrt greedy length:", len(path))
    traj_out = []
    i=0
    while i <= len(path) - 2:
        traj_part, _ = trajectory(path[i], path[i + 1], type = 1)
        traj_out = traj_out + traj_part
        i = i + 1
    return traj_out

if __name__ == '__main__':

    obstaclelist=[(200, 500, 20, 100),(200, 1000, 20, 100),(600, 900 , 20, 100),(900, 0, 20, 500),(900, 1000, 20, 500),(1400, 0, 20, 100),(1400, 1400, 20, 100)]
    initial_position = np.array([100, 800, 0])
    target_position = np.array([1250, 60, -np.pi/2])
    #target_position = np.array([1200, 1400, 0])
    ospace = Ospace(initial_position, target_position, obstaclelist)
    cspace = Cspace(initial_position, target_position, obstaclelist)
    random_boundary = [(0, 1500), (0, 1500), (-np.pi, np.pi)]
    traj = rrt_planner(initial_position, target_position,random_boundary,cspace)
    draw_car(traj,obstaclelist)
    traj = rrt_greedy(initial_position, target_position, random_boundary, cspace)
    draw_car(traj,obstaclelist)



