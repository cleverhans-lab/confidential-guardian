import matplotlib.pyplot as plt

# empirically verifying the behavior of the widgets defined in the proof

# constants
EPS_CLIP_GLOBAL_DEFAULT = 0.1 # 
EPS_LB_GLOBAL_DEFAULT = 1 # distance from t that does not get 0'd out in LB
EPS_UB_GLOBAL_DEFAULT = 1 # distance from t that does not get 0'd out in UB
EPS_AND_GLOBAL_DEFAULT = 0.1 # 


# x a scalar
def ReLU(x):
    if (x<=0):
        return 0
    else:
        return x
    

def clipped_lb_widget(x, t, eps_lb=EPS_LB_GLOBAL_DEFAULT, eps_clip=EPS_CLIP_GLOBAL_DEFAULT):
    return ReLU(ReLU(x)-(t-eps_lb)) - ReLU(ReLU(x-eps_clip)-(t-eps_lb))

def clipped_ub_widget(x, t, eps_ub=EPS_UB_GLOBAL_DEFAULT, eps_clip=EPS_CLIP_GLOBAL_DEFAULT):
    return ReLU(-ReLU(x) + (t+eps_ub)) - ReLU(-ReLU(x+eps_clip)+(t + eps_ub))

def and_widget(o1, o2, eps_and=EPS_AND_GLOBAL_DEFAULT, eps_clip=EPS_CLIP_GLOBAL_DEFAULT):
    return ReLU((o1 + o2) - (2 * eps_clip - eps_and))


def plot_ub():
    xs = [x*0.1 for x in range(-50,51)]
    ys = [clipped_ub_widget(x, 2.5) for x in xs]
    plt.plot(xs, ys)
    plt.show()

def plot_lb():
    xs = [x*0.1 for x in range(-50,51)]
    ys = [clipped_lb_widget(x, 2.5) for x in xs]
    plt.plot(xs, ys)
    plt.show()

def plot_and():
    o1s = []
    o2s = []
    rets = []
    for x in range(0, 11):
        for y in range(0, 11):
            o1 = x*0.01
            o2 = y*0.01
            ret = and_widget(o1, o2)
            o1s.append(o1)
            o2s.append(o2)
            rets.append(ret)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(o1s, o2s, rets)
    ax.set_xlabel("Input 1")
    ax.set_ylabel("Input 2")
    ax.set_zlabel("AND widget output")
    plt.show()

