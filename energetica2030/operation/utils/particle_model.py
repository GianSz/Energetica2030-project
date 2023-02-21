def bike_model_particle(hybrid_cont, speeds, slopes,last_vel,hev):    
    p = []
    p1 = []
    p2 = []


    vel = speeds/3.6
    theta = slopes*math.pi/180
    faero = 0.5 * hev.Ambient.rho * hev.Chassis.a * hev.Chassis.cd * (vel ** 2)
    # Rolling force
    froll = hev.Ambient.g * hev.Chassis.m * hev.Chassis.crr * np.cos(theta)
    fg = hev.Ambient.g * hev.Chassis.m * np.sin(theta)
    delta_v = vel - last_vel
    #print(vel,last_vel)
    f_inertia = hev.Chassis.m * delta_v / 1
    # Sum Forces
    fres = faero + froll + fg + f_inertia

    # calculos Mauro
    p_e = vel * fres
    p_m = (fres * hev.Wheel.rw) * (vel / hev.Wheel.rw)
    #if abs(delta_v) < 1:
    p_eb = p_m * (1 - hybrid_cont) / 0.7
    p_cn = p_m * hybrid_cont/0.2

    #p_em = (p_m * hybrid_cont) / 0.2
    #Cc = p_em / ((speeds[i]/3600)*i)
    #Ccombustible = (Cc)/(33.7)
    # else:
    #p_eb = p_e / (p_m * 0.7)  # 0.7 es la eficiencia (estimada) del tren motriz
    p.append(p_eb)
    p2.append(p_cn)
    p1.append(p_m)
    #p1.append(p_em)
    #p2.append(Cc)
    #p3.append(Ccombustible)
    last_vel = vel

    pow_consumption = sum(p)/3600  # watts hour
    pcn_consumption = sum(p2)/3600
    p_consumption = sum(p1)
    #pow_consumption_comb = sum(p1)/3600  # watts hour

    seconds = len(p)
    result = pow_consumption
    result1 = pcn_consumption
    result2 =p_consumption
    #result1 = State_bater - pow_consumption
    #result1 = (pow_consumption_comb, seconds)
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(3)
    # axs[0].plot([i for i in range(len(speeds))], p)
    # axs[0].set_title('Travel Velocity')
    # plt.show()
    #State_bater = result1
    return result, p_m,result1,result2,last_vel,hev

Spped=[]
slope=[]
delta=[]
P1 = []
P2 = []
P3 = []
State_bater = 100
last_vel = 0
hybrid_cont = 0.4
import numpy as np
import math
if hybrid_cont == 0:
    from parameters_electric import HEV
else:
    from parameters_hybrid import HEV
hev = HEV()


for i in range(3600):
    Spped.append(60)
    delta.append(3600)
    slope.append(0)
    R1, R2, R3, R4, last_vel,hev = bike_model_particle(hybrid_cont, Spped[i], slope[i],last_vel,hev)
    P1.append(R1)
    P2.append(R3)
    P3.append(Spped[i]/delta[i])
    #print(bike_model_particle(0.7, Spped[i], slope[i],State_bater))
    print(R1)
    #print(R2) 
    print(R3)
    #print(R4)

print(sum(P1))
print(sum(P2))
#print(sum(P3))

#R1, R2 = bike_model_particle(0.4, Spped, slope)
#print(R1)
#print(R2)   

'''for i in range(5):
    Spped.append(60)
    slope.append(0)
    R1, R2, State_bater = bike_model_particle(0.4, Spped[i], slope[i],State_bater)
    #print(bike_model_particle(0.7, Spped[i], slope[i],State_bater))
    print(State_bater)
    print(R1)
    print(R2)'''
