from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import traci
import quadprog
import pickle

from energetica2030.settings import UTILS_PATH
from energetica2030.settings import STATIC_OPERATION_PATH

route_dict = {}
new_list = []
motOBJ_dic = {}
m_list = []
m_1_list = []
m_2_list = []
pkID = []
parkOBJ_dic = {}
stID = []
statOBJ_dic = {}
stat_ship_OBJ_dic = {}
route_dict_ship = {}
new_list_ship = []
parkshipOBJ_dic = {}
shipOBJ_dic = {}
ch_ship = 0
demanss_1=[]


# SIMULACIÓN
tsim = 0
step = 0
Demandas = 0
Demandaship = 0
#Demandas = np.zeros([tsim, 1], dtype = float)
#Demandaship = np.zeros([tsim, 1], dtype = float)
velocidad_ship = []

@login_required(login_url='/logIn/')
#This function renders the operation page
def operationPage(request):
    global tsim
    global Demandas
    global Demandaship
    if(request.method == 'GET'):
        try:
            route = request.GET["selectRoute"]
            if(int(route) == 1):
                tsim = 7000
            else:
                tsim = 2000
            Demandas = np.zeros([tsim, 1], dtype = float)
            Demandaship = np.zeros([tsim, 1], dtype = float)

            if(route != '0'):
                operationExecution(step, m_list, ch_ship, int(route))
                return render(request, 'operation/operationPage.html', context={'graphics':True})
            else:
                return render(request, 'operation/operationPage.html', context={'graphics':False})
        except:
            return render(request, 'operation/operationPage.html', context={'graphics':False})

def QP(Demanda_a):
    #DATA ORGANIZADA
    
    # Parámetros  tp >= Np
    Vp = 2 #Ventana de predicción
    Np = 2 #horizonte de predicción   
    tp = 24#tiempo de simulación horas
    Nc = 3 #Horizonte de control = # de entradas (3 entradas) 
    radiationa = pd.read_csv(UTILS_PATH+'/GHI1.csv', sep=',', header=None, index_col=None, usecols=[7])
    radiationb = pd.read_csv(UTILS_PATH+'/GHI2.csv', sep=',', header=None, index_col=None, usecols=[7])
    radiationc = pd.read_csv(UTILS_PATH+'/GHI3.csv', sep=',', header=None, index_col=None, usecols=[7])

    radiation = pd.concat([radiationa, radiationb, radiationc], ignore_index=True)

    with open(UTILS_PATH+"/DATA.pickle", "rb") as f:
        undia = pickle.load(f)

    with open(UTILS_PATH+"/price.pickle", "rb") as f:
        price = pickle.load(f)

    Dem = np.zeros(tp+Np-1)
    jj = 0
    ij = 1
    for ii in range(0, len(undia), 3600):

        Dem[jj] = sum(undia[ii:3600*ij])*3700

        jj = jj + 1
        ij = ij + 1

    #Dem = Dem + Dem
    def quadprog_solve_qp(P, q, G_p, g_p, A_p, a_p):
        qp_G = 0.5 * (P + P.T) + np.identity(len(P))*1e-5  # make sure P is symmetric
        qp_a = -q #-q
        if A_p is not None:
            qp_C = -np.vstack([A_p, G_p]).T
            qp_b = -np.hstack([a_p, g_p])
            meq = A_p.shape[0]
        else:  # no equality constraint
            qp_C = -G_p.T
            qp_b = -g_p
            meq = 0
        return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


    RAD =[]
    ik = 0
    for kk in range(int(len(radiation)/2)):
        RAD.append(radiation[7][ik])

        ik = ik + 2



    #radiation1 = np.array(RAD[0:tp+Np-1]);
    #radiation1 = np.array(RAD[0:tp+Np-1]);
    radiation1 = np.array(RAD[tp+Np-2:tp+Np +23 ]);

    #radiation2 = radiation1[24:]
    #Precio = pd.read_csv('DATA1.csv', sep=';', header=0, index_col=None, usecols=[2]);
    #Precio = np.array(Precio[0:tp+Np-1]);
    E_f_1 = radiation1*16 # 16 metros cuadrados
    E_f = []
    for ii in range(len(E_f_1)):
        E_f.append(E_f_1[ii])


    '''Dem1 = np.zeros(tp+Np-1)
    D=Dem1

    D[11] = 42470
    D[13] = 44330
    D[15] = 91580
    D[17] = 44330'''

    E_f =np.array(E_f)
    D = Demanda_a #*0.3#sin_wave(AA=100, ff=5, fs=tp+Np-1, phi=0, ttt=1)tsim

    p1 = np.ones(tp+Np-1)*500*0.00022 #np.array(price[0:tp+Np-1]) #np.ones(tp+Np-1)*500
    p2 = np.array(price[0:tp+Np-1])*0.00022 #np.ones(tp+Np-1)*500
    #p2[int((tp+Np-1)/2):] = np.array(price[0:tp+Np-1]) #np.ones(p2[10:].shape[0])*500
    #E_f = np.ones(tp+Np-1)*120

    # Modelo energía en la batería
    A_m = np.ones(1)
    B_m = np.zeros((1,3))
    B_m[0][0] = 1
    X_p = np.zeros(tp+1)
    X_p [0:1]= [12000]

    #Condiciones iniciales
    X_m = np.zeros(1)
    X_m [0]= 12000
    w_m = D
    U_m = np.zeros(tp*Nc)

    # Predicción de la salida
    Y = np.zeros(tp+1)
    Y [0:1]= [0]
    YY = np.zeros(tp+1)
    YY [0:1]= [0]
    B_p = np.zeros((1,3))
    w_p = np.zeros(1)
    Q = np.zeros((Np,Np*Nc))
    W = np.zeros((Np,Np))
    D_d = np.zeros((Np,Np*Nc))

    #Creación de matrices de restricciones
    A_p = np.zeros((Np,Np*Nc))
    a_p = E_f[:]- D[0:len(E_f)]   # E_f[:,0]
    G_p = np.zeros((4*Np,Np*Nc))
    g_p = np.zeros(4*Np)
    C_res = np.ones(1)
    C_res[0] = -1
    U_o = np.zeros(Nc)

    j=0
    for k in range(Np):
        A_p[k,j:j+Nc] = [1,1,1]
        j = j+3

    j=0
    for k in range(2*Np,3*Np,1):
        G_p[k,j:j+Nc] = [0,-1,0]
        j = j+3

    j=0
    for k in range(3*Np,4*Np,1):
        G_p[k,j:j+Nc] = [0,1,0]
        j = j+3


    for k in range(3*Np,4*Np,1):
        g_p[k] = 1
        

    v=0
    for ll in range(0,Np*Nc,Nc):
            D_d[v,ll:ll+3] = [0,1,0]
            v=v+1

    #Quadprog Optimización
    m=0
    r = 1
    for i in range(tp): 
        kk = 0
        jj = 0
        for k in range(0,Np*Nc,Nc):
            for j in range(Np):
                if jj >= kk :
                    G_p[j,k:k+Nc] = [-1,0,0]
                elif kk-jj==1:
                    G_p[j,k:k+Nc] = 0
                else:
                    G_p[j,k:k+Nc] = 0
                jj = jj+1
            kk=kk+1
            jj = 0

        kk = 0
        jj = 0
        for k in range(0,Np*Nc,Nc):
            for j in range(Np,2*Np):
                if jj >= kk :
                    G_p[j,k:k+Nc] = [1,0,0]
                elif kk-jj==1:
                    G_p[j,k:k+Nc] = 0
                else:
                    G_p[j,k:k+Nc] = 0
                jj = jj+1
            kk=kk+1
            jj = 0


        j=0
        jj=0
        for k in range(0,Np,1):
            B_p[0][1] = -p1[jj+i]
            B_p[0][2] = p2[jj+i]
            w_p[0] = p1[jj+i]
            Q[k,j:j+Nc] = B_p
            W[k,k] = w_p
            j = j+3
            jj=jj+1

        for k in range(0,Np):
            g_p[k] = X_m[0]
        
        for k in range(Np,2*Np):
            g_p[k] = 12000 - X_m[0]

        P = np.matmul(D_d.T, D_d) + np.matmul(Q.T,Q) #  np.matmul(D_d.T, D_d) + 
        q= np.matmul(-Q.T, np.matmul(W, w_m[i:i+Np]))
        u = quadprog_solve_qp(P, q, G_p, g_p, A_p, a_p[i:i+Np])


        
        U_m[m:m+3] = u[0:3]
        X_p[r:r+1] = np.matmul(A_m, X_m) + np.matmul(B_m, U_m[m:m+3])
        X_m = X_p[r:r+1]
        Y[r:r+1] =  np.matmul(Q[0],u) + np.matmul(W[0], w_m[i:i+Np])

        if i == 2:
            a=1
        
        m=m+3
        r=r+1

    j=0
    jj=0
    QQ = np.zeros((tp,tp*Nc))
    WW = np.zeros((tp,tp))
    for k in range(0,tp,1):
        B_p[0][1] = -p1[jj]
        B_p[0][2] = p2[jj]
        w_p[0] = p1[jj]
        QQ[k,j:j+Nc] = B_p
        WW[k,k] = w_p
        j = j+3
        jj=jj+1

    YY[1:] =  np.matmul(QQ, U_m) + np.matmul(WW, w_m[0:tp])

    #Separación de variables de decisión
    error = np.zeros(tp)
    E_sp = np.zeros(tp)
    E_bc = np.zeros(tp)
    E_bcc = np.zeros(tp)
    E_b = np.zeros(tp+1)
    G = np.zeros(tp+1)

    er = 0
    for g in range(1,tp*Nc,3):
        error[er] = U_m[g]
        er = er+1

    er = 0
    err = 0
    for g in range(0,tp*Nc,3):
        E_bc[er] = U_m[g]
        E_bcc[err] = U_m[g]
        er = er+1
        err = err+1

    er = 0
    for g in range(2,tp*Nc,3):
        E_sp[er] = U_m[g]
        er = er+1

    E_b = X_p

    G = Y

    t = np.arange(0,tp,1)
    tt = np.arange(0,tp+1,1)

    Result1 = E_f[0:tp]-E_bc[0:tp]-E_sp[0:tp]
    Result2 = G[0:tp]
    Result3 = E_b
    Result4 = D[0:tp]
    Result5 = E_bc
    Result6 = E_f[0:tp]
    Result7 = E_sp
    Result8 = p2

    return Result1, Result2, Result3, Result4, Result5, Result6, Result7, Result8, tt, t

class moto:

    def __init__(self, moto_id_in, Parking_in):

        self.moto_id = moto_id_in
        self.Parking = Parking_in
        self.originEdge = []
        self.destinationEdge = []
        self.new_route_1 = []
        self.new_route_2 = []
        self.new_route = []
        self.tripstate = []
        #self.speed= []
        self.speed = np.zeros([tsim, 1], dtype = float)
        self.acel= []
        self.hybrid_cont= []
        self.new_route_2 = []
        self.new_route_n = []
        self.last_vel= 0
        self.trip_c = 0
        self.State_bater = 3700
        self.Energy= np.zeros([tsim, 1], dtype = float)
        self.Energy1= np.zeros([tsim, 1], dtype = float)
        self.combus= np.zeros([tsim, 1], dtype = float)
        self.combus2= np.zeros([tsim, 1], dtype = float)
        self.E_TOTAL= np.zeros([tsim, 1], dtype = float)
        self.moto_dist2 = np.zeros([2, 1], dtype = int)
        self.pow_consumption = []
        self.pcn_consumption = []
        self.moto_dist1 = np.zeros([2, 1], dtype = int)

    def viaje_OD(self, originEdge, destinationEdge):

        self.originEdge = originEdge
        self.destinationEdge = destinationEdge
        #Origen
        current_route = traci.lane.getEdgeID(traci.parkingarea.getLaneID(self.Parking))
        CP1 =traci.simulation.findRoute(current_route,self.originEdge).edges
        self.new_route = list(CP1)

        traci.vehicle.setRoute(self.moto_id, self.new_route)                             # Set new route for the vehicle
        traci.vehicle.setParkingAreaStop(self.moto_id, self.Parking, 0)          # Getting out of parking area
        origin_lane = originEdge + "_0"
        length_lane = traci.lane.getLength(origin_lane)                         # Getting the length of the destination lane
        traci.vehicle.setStop(self.moto_id, originEdge, length_lane / 2, 0, 10)     # Vehicle will stop in the middle of origin edge (VehID, Edge, PosStop, LaneStop, TimeStop)
        # Destination
        CP2 =traci.simulation.findRoute(self.originEdge,self.destinationEdge).edges
        CP = CP1+CP2[1:]
        self.new_route_1 = list(CP)
        traci.vehicle.setRoute(self.moto_id, self.new_route_1)                             # Set new route for the vehicle
        destination_lane = destinationEdge + "_0"
        length_lane = traci.lane.getLength(destination_lane)
        traci.vehicle.setStop(self.moto_id, destinationEdge, length_lane / 2, 0, 10) 
        self.Parking = ''

    def viaje_stat1(self, parking_in):      
        lane_id = traci.parkingarea.getLaneID(parking_in)
        Edge_id = traci.lane.getEdgeID(lane_id)
        CP6 = traci.simulation.findRoute(traci.lane.getEdgeID(traci.vehicle.getLaneID(self.moto_id)),Edge_id).edges
        self.new_route_2 = list(CP6)
        traci.vehicle.setRoute(self.moto_id, self.new_route_2)                             # Set new route for the vehicle
        length_lane = traci.lane.getLength(self.new_route_2[len(self.new_route_2)-1]+ "_0")
        traci.vehicle.setStop(self.moto_id, self.new_route_2[len(self.new_route_2)-1], length_lane / 3, 0, 5) 
        #traci.vehicle.setParkingAreaStop(self.moto_id, parking_in) 
        self.Parking = parking_in

    def viaje_stat(self, parking_in):      
        lane_id = traci.parkingarea.getLaneID(parking_in)
        Edge_id = traci.lane.getEdgeID(lane_id)
        CPp = traci.simulation.findRoute(traci.lane.getEdgeID(traci.vehicle.getLaneID(self.moto_id)),Edge_id).edges
        self.new_route_2 = list(CPp)
        traci.vehicle.setRoute(self.moto_id, self.new_route_2)                             # Set new route for the vehicle
        length_lane = traci.lane.getLength(self.new_route_2[len(self.new_route_2)-1]+ "_0")
        traci.vehicle.setStop(self.moto_id, self.new_route_2[len(self.new_route_2)-1], length_lane / 3, 0) 
        self.Parking = parking_in

    def v_stat_park(self, p_ing):      
        lane_id = traci.parkingarea.getLaneID(p_ing)
        Edge_id = traci.lane.getEdgeID(lane_id)
        lane_id_0 = traci.parkingarea.getLaneID(self.Parking)
        Edge_id_0 = traci.lane.getEdgeID(lane_id_0)
        CPpp = traci.simulation.findRoute(Edge_id_0,Edge_id).edges
        self.new_route_n = list(CPpp)
        traci.vehicle.setRoute(self.moto_id, self.new_route_n)                             # Set new route for the vehicle
        traci.vehicle.setParkingAreaStop(self.moto_id, self.Parking, 0)          # Getting out of parking area
        traci.vehicle.setParkingAreaStop(self.moto_id, p_ing)
        self.Parking = p_ing

    def getin_stat1(self,park_stat): 
        traci.vehicle.setParkingAreaStop(self.moto_id, park_stat)
        self.Parking = park_stat

    def getin_stat(self,park_stat): 
        length_lane = traci.lane.getLength(self.new_route_2[len(self.new_route_2)-1]+ "_0") 
        traci.vehicle.setStop(self.moto_id, self.new_route_2[len(self.new_route_2)-1], length_lane / 3, 0, 1) 
        traci.vehicle.setParkingAreaStop(self.moto_id, park_stat)
        self.Parking = park_stat

    def moto_state(self):
        lane_moto_0 = traci.vehicle.getLaneID(self.moto_id)
        stop_state = traci.vehicle.getStopState(self.moto_id)
        if stop_state == 131:
            tr = 0
        elif stop_state == 0:
            tr = 1
        elif traci.lane.getEdgeID(lane_moto_0) == self.destinationEdge: 
            tr = 3
        elif stop_state == 1:
            tr = 2     
        self.tripstate.append(tr)

        return self.tripstate, tr # 0:parked, 1:on trip, 2:Stop (origen), 3: trip finished(destino)

    def moto_peti(self,trip_c):
        self.trip_c = trip_c
        return self.trip_c

    def E(self,moto_id_in,trips,step):
        self.speed[step] = traci.vehicle.getSpeed(moto_id_in) #self.speed.append(traci.vehicle.getSpeed(moto_id_in))
        self.acel.append(traci.vehicle.getAcceleration(moto_id_in))

        if self.Energy[step-1] >= 3430 and (moto_id_in not in traci.parkingarea.getVehicleIDs("m3")):
            if trips == 1:
                    
                self.hybrid_cont = 0.4
                if self.hybrid_cont == 0:
                    from operation.utils.parameters_electric import HEV
                else:
                    from operation.utils.parameters_hybrid import HEV
                self.hev = HEV()
            

                real_speed = traci.vehicle.getSpeed(moto_id_in)
                slopes = 0

                vel = real_speed
                theta = slopes*math.pi/180
                faero = 0.5 * self.hev.Ambient.rho * self.hev.Chassis.a * self.hev.Chassis.cd * (vel ** 2)

                froll = self.hev.Ambient.g * self.hev.Chassis.m * self.hev.Chassis.crr * np.cos(theta)
                fg = self.hev.Ambient.g * self.hev.Chassis.m * np.sin(theta)
                delta_v = vel - self.last_vel

                f_inertia = self.hev.Chassis.m * delta_v / 1

                fres = faero + froll + fg + f_inertia
                p_e = vel * fres
                p_m = (fres * self.hev.Wheel.rw) * (vel / self.hev.Wheel.rw)
                p_eb = p_m * (1 - self.hybrid_cont) / 0.7
                p_cn = p_m * (self.hybrid_cont) / 0.2

                if p_cn <= 0:
                    p_cn = 0


                self.last_vel = vel

                self.pow_consumption = p_eb/3600  # watts hour
                self.pcn_consumption = p_cn/3600  # watts hour

                result = self.pow_consumption
                result1 = self.State_bater - self.pow_consumption  #self.State_bater - 
                result2 = self.pcn_consumption / 36718.50158230244 #galones
                result3 = self.pcn_consumption 
                self.Energy[step] = result1 
                self.Energy1[step] = result
                self.combus[step] = result2
                self.combus2[step] = result3
                self.E_TOTAL[step] = self.pow_consumption + self.pcn_consumption
                self.State_bater = result1


            if trips == 2 or trips == 3:
                self.Energy[step]= self.Energy[step-1] 
                self.State_bater = self.Energy[step]
                self.combus[step] = 0
                self.E_TOTAL[step] = 0  
                self.Energy1[step] = 0

            if moto_id_in in traci.parkingarea.getVehicleIDs("m1") or moto_id_in in traci.parkingarea.getVehicleIDs("m2"):      
                self.Energy[step]= self.Energy[step-1]
                self.State_bater = self.Energy[step]
                self.combus[step] = 0
                self.E_TOTAL[step] = 0  
                self.Energy1[step] = 0

        elif self.Energy[step-1] < 3430 and (moto_id_in not in traci.parkingarea.getVehicleIDs("m3")):
            if trips == 1:
                    
                self.hybrid_cont = 1
                if self.hybrid_cont == 0:
                    from operation.utils.parameters_electric import HEV
                else:
                    from operation.utils.parameters_hybrid import HEV
                self.hev = HEV()
            

                real_speed = traci.vehicle.getSpeed(moto_id_in)
                slopes = 0

                vel = real_speed
                theta = slopes*math.pi/180
                faero = 0.5 * self.hev.Ambient.rho * self.hev.Chassis.a * self.hev.Chassis.cd * (vel ** 2)

                froll = self.hev.Ambient.g * self.hev.Chassis.m * self.hev.Chassis.crr * np.cos(theta)
                fg = self.hev.Ambient.g * self.hev.Chassis.m * np.sin(theta)
                delta_v = vel - self.last_vel

                f_inertia = self.hev.Chassis.m * delta_v / 1

                fres = faero + froll + fg + f_inertia
                p_e = vel * fres
                p_m = (fres * self.hev.Wheel.rw) * (vel / self.hev.Wheel.rw)
                p_eb = p_m * (1 - self.hybrid_cont) / 0.7
                p_cn = p_m * (self.hybrid_cont) / 0.2

                if p_cn <= 0:
                    p_cn = 0


                self.last_vel = vel

                self.pow_consumption = p_eb/3600  # watts hour
                self.pcn_consumption = p_cn/3600  # watts hour

                result = self.pow_consumption
                result1 = self.State_bater - self.pow_consumption  #self.State_bater - 
                result2 = self.pcn_consumption / 36718.50158230244 #galones
                result3 = self.pcn_consumption 
                self.Energy[step] = result1 
                self.Energy1[step] = result
                self.combus[step] = result2
                self.combus2[step] = result3
                self.E_TOTAL[step] = self.pow_consumption + self.pcn_consumption
                self.State_bater = result1

            if trips == 2 or trips == 3:
                self.Energy[step]= self.Energy[step-1] 
                self.State_bater = self.Energy[step]
                self.combus[step] = 0
                self.E_TOTAL[step] = 0  
                self.Energy1[step] = 0

            if moto_id_in in traci.parkingarea.getVehicleIDs("m1") or moto_id_in in traci.parkingarea.getVehicleIDs("m2"):      
                self.Energy[step]= self.Energy[step-1]
                self.State_bater = self.Energy[step]
                self.combus[step] = 0
                self.E_TOTAL[step] = 0  
                self.Energy1[step] = 0
  
        if self.Energy[step-1] >= 3430 and (moto_id_in in traci.parkingarea.getVehicleIDs("m3")) and (self.Parking == 'm1' or self.Parking == 'm2'):
            self.Energy[step]= self.Energy[step-1] 
            self.State_bater = self.Energy[step]
            self.combus[step] = 0
            self.E_TOTAL[step] = 0  
            self.Energy1[step] = 0

        return self.Energy, self.combus, self.Energy1 

    def m_dist(self,moto_id_in,moto_dist_id):
        
        idis = traci.vehicle.getLaneID(moto_id_in)
        l_idis = len(idis)
        k=0

        if moto_id_in in traci.parkingarea.getVehicleIDs("m1") and idis == '':
            idis = traci.parkingarea.getLaneID("m1")
            
        if moto_id_in in traci.parkingarea.getVehicleIDs("m2") and idis == '':
            idis = traci.parkingarea.getLaneID("m2") 

        for i in moto_dist_id:
            if '_0' in idis:
                self.moto_dist1[k] = int(traci.simulation.getDistanceRoad(idis[0:l_idis-2],0, i, 0, isDriving=True))
            elif '_1' in idis:
                self.moto_dist1[k] = int(traci.simulation.getDistanceRoad(idis[0:l_idis-2],0, i, 0, isDriving=True))
            else:
                self.moto_dist1[k] = int(traci.simulation.getDistanceRoad(idis,0, i, 0, isDriving=True))
            k += 1

        return self.moto_dist1 

    def m_dist_park(self,moto_dist_id,St):
        idis = traci.parkingarea.getLaneID(St)
        l_idis = len(idis)
        k=0
        for i in moto_dist_id:
            self.moto_dist2[k] = int(traci.simulation.getDistanceRoad(idis[0:l_idis-2],0, i, 0, isDriving=True))
            k = k+1

        return self.moto_dist2

class ship:
    def __init__(self, ship_in, Parkingship_in):

        self.ship_id = ship_in
        self.Parkingship = Parkingship_in
        self.originEdgeship = []
        self.destinationEdgeship = []
        self.tripstateship = []  # 0:parked, 1:on trip, 2: trip finished, 3: to parking
        self.Energyship = []
        self.ship_id
        self.P_i = int(92*4*3.7*87) #(118459.20000000001)
        self.pot = np.zeros([tsim, 1], dtype = int) 
        self.S_E = np.zeros([tsim, 1], dtype = int)  
        self.sspeed = np.zeros([tsim, 1], dtype = int)  

    def v_spark_spark(self, p_ing):      
        
        lane_id = traci.parkingarea.getLaneID(p_ing)
        Edge_id = traci.lane.getEdgeID(lane_id)
        traci.vehicle.changeTarget(self.ship_id, Edge_id)                        # Create a new route to destination edge
        traci.vehicle.setParkingAreaStop(self.ship_id, self.Parkingship, 0)          # Getting out of parking area
        length_lane = traci.lane.getLength(lane_id)                         # Getting the length of the destination lane
        traci.vehicle.setStop(self.ship_id, Edge_id, length_lane / 2, 0, 10)     # Vehicle will stop in the middle of origin edge (VehID, Edge, PosStop, LaneStop, TimeStop)
        traci.vehicle.setParkingAreaStop(self.ship_id, p_ing)
        self.Parkingship = p_ing
    
    def v_spark_spark_1(self, p_ing):      
        
        lane_id = traci.parkingarea.getLaneID(p_ing)
        Edge_id = traci.lane.getEdgeID(lane_id)
        traci.vehicle.changeTarget(self.ship_id, Edge_id)                        # Create a new route to destination edge
        traci.vehicle.setParkingAreaStop(self.ship_id, self.Parkingship, 0)          # Getting out of parking area
        length_lane = traci.lane.getLength(lane_id)                         # Getting the length of the destination lane
        traci.vehicle.setStop(self.ship_id, Edge_id, length_lane / 2, 0, 10)     # Vehicle will stop in the middle of origin edge (VehID, Edge, PosStop, LaneStop, TimeStop)
        self.Parkingship = p_ing

    def viaje_pship(self, parkingship_in):

        current_route = traci.vehicle.getRoute(self.ship_id)                        # Getting the current rout for the vehicle
        new_route = []
        lane_id = traci.parkingarea.getLaneID(parkingship_in)
        Edge_id = traci.lane.getEdgeID(lane_id)

        for r in current_route:
            new_route.append(r)
        new_route.append(Edge_id)                                       # List of new route appending the destination edge at the end.

        traci.vehicle.setRoute(self.ship_id, new_route)                             # Set new route for the vehicle
        traci.vehicle.rerouteEffort(self.ship_id)                                 # Compute the new route
        traci.vehicle.setParkingAreaStop(self.ship_id, parkingship_in)

        self.tripstateship = 1

    def ship_state(self):

        lane_ship_0 = traci.vehicle.getLaneID(self.ship_id)
        stop_state = traci.vehicle.getStopState(self.ship_id)

        if stop_state == 131: # 131 means moto is in parking
            self.tripstateship = 0
        elif stop_state == 0:
            self.tripstateship = 1

        return self.tripstateship

    def E_ship(self,ship_in,indic,step):
        n_d = 0.75
        n_m = 0.95
        n_t = 0.931
        n_p = 0.98

        #if ship_in in traci.parkingarea.getVehicleIDs("s4") or ship_in in traci.parkingarea.getVehicleIDs("s5") or ship_in in traci.parkingarea.getVehicleIDs("s6") or indic == 0:

        #    self.S_E[step] = self.P_i

        if step <= 100:
            self.S_E[step] = self.P_i

        if indic == 0:
            self.S_E[step] = self.S_E[step-1]

            
        if indic == 1:
            vel_ship = traci.vehicle.getSpeed(ship_in)
            P_e = (1.4*vel_ship**(4)-66.3*vel_ship**(3)+1109.4*vel_ship**(2)+452.6*vel_ship)/(n_d*n_m*n_t*n_p) # Potencia eléctrica en W
            self.pot[step] = P_e  #[kW]
            self.S_E[step] = self.P_i - (P_e/3600)
            self.P_i = self.S_E[step]
    
        return self.S_E

    def vel_bus(self,ship_in):

        vel_ship = traci.vehicle.getSpeed(ship_in) 
        #Devuelve el consumo de electricidad en Wh / s para el último paso de tiempo.
        #Multiplique por la longitud del paso para obtener el valor
        self.sspeed[step] = vel_ship

class pconcent:

    def __init__(self,parkingID):

        self.pkID = pkID
        #self.Asig = 0

    def count(self,pkID):
        
        countpark = traci.parkingarea.getVehicleIDs(pkID)
        free = 10 - len(countpark) #- self.Asig
        return countpark,free 

class stat_charg:

    def __init__(self,stat_id):

        self.stID = stID
        
    def count(self,stID):
        countstat = traci.parkingarea.getVehicleIDs(stID)
        free = 10 - len(countstat)
        return countstat , free

    def charge(self,mt,step):

        if (motOBJ_dic[mt].Energy[step-1] + 1) < 3700:
           motOBJ_dic[mt].Energy[step] = motOBJ_dic[mt].Energy[step-1] + 1
           motOBJ_dic[mt].State_bater= motOBJ_dic[mt].Energy[step]
        elif (motOBJ_dic[mt].Energy[step-1] + 1) >= 3700:
           motOBJ_dic[mt].Energy[step] = 3700
           motOBJ_dic[mt].State_bater = motOBJ_dic[mt].Energy[step]

class stat_charg_ship:

    def __init__(self,stat_id):

        self.stID = stID
        
    def count(self,stID):
        countstat = traci.parkingarea.getVehicleIDs(stID)
        free = 10 - len(countstat)
        return countstat , free

    def charge(self,m_s,step):

        if (shipOBJ_dic[m_s].S_E[step-1] + 100) < int(92*4*3.7*87):
           shipOBJ_dic[m_s].S_E[step] = shipOBJ_dic[m_s].S_E[step-1] + 100
           shipOBJ_dic[m_s].P_i= shipOBJ_dic[m_s].S_E[step]
        elif (shipOBJ_dic[m_s].S_E[step-1] + 100) >= int(92*4*3.7*87):
           shipOBJ_dic[m_s].S_E[step] = int(92*4*3.7*87)
           shipOBJ_dic[m_s].P_i = shipOBJ_dic[m_s].S_E[step]

class pshipconcent:

    def __init__(self,parkingIDship):

        self.pkIDship = parkingIDship

    def count(self,parkingIDship):
        
        countpark = traci.parkingarea.getVehicleIDs(parkingIDship)
        free = 10 - len(countpark)
        return countpark,free

def startparking(parkingID, motoNumber):

    count_moto = 0
    count_route = 300

    for p in range(len(parkingID)):
        p_i = parkingID[p]                                          # ParkingID p
        n_i = motoNumber[p]                                         # Initial motorcycles in parking p
        lane_PA = traci.parkingarea.getLaneID(p_i)                  # Lane of parking p
        edge_PA = traci.lane.getEdgeID(lane_PA)                     # Cdge of parking p
        route_id = "route_" + str(count_route)
        route_dict[p_i] = route_id

        # Create routeID
        traci.route.add(route_id, [edge_PA])                        # Create route starting in parking edge
        for m in range(n_i):
            id_moto = "moto_" + str(count_moto)                     # create motorcycle ID (must be unique
            traci.vehicle.add(id_moto, route_id, typeID="moto")     # Add motorcycle to simulation given a route and vehicle type
            traci.vehicle.setParkingAreaStop(id_moto, p_i)          # Setting in a motorcycle into a parkingArea
            new_list.append(id_moto)
            count_moto += 1
        count_route += 1

def startparkingship(parkingIDship, shipNumber):

    count_ship = 0
    count_route = 100

    for p in range(len(parkingIDship)):
        p_i = parkingIDship[p]                                          # ParkingID p
        n_i = shipNumber[p]                                         # Initial motorcycles in parking p
        lane_PA = traci.parkingarea.getLaneID(p_i)                  # Lane of parking p
        edge_PA = traci.lane.getEdgeID(lane_PA)                     # Cdge of parking p
        route_id_ship = "route_" + str(count_route)
        route_dict_ship[p_i] = route_id_ship

        # Create routeID
        traci.route.add(route_id_ship, [edge_PA])                        # Create route starting in parking edge
        for m in range(n_i):
            id_ship = "ship_" + str(count_ship)                    # create motorcycle ID (must be unique
            traci.vehicle.add(id_ship, route_id_ship, typeID="ship")     # Add motorcycle to simulation given a route and vehicle type
            traci.vehicle.setParkingAreaStop(id_ship, p_i)          # Setting in a motorcycle into a parkingArea
            new_list_ship.append(id_ship)
            count_ship += 1
        count_route += 1

def operationExecution(step, m_list, ch_ship, route):
    step = step
    m_list = m_list
    ch_ship = ch_ship
    #traci.start(["sumo-gui", "-c", UTILS_PATH+"/osm1.sumocfg", "--start"])
    traci.start(["sumo","-c", UTILS_PATH+"/osm1.sumocfg", "--quit-on-end"])
    parkingID = ["m1","m2"]
    parkingID_1 = ["372051587#0", "-111215443#2"]
    motoNumber = [10,5]
    stat_char_1 = ['-400299303#12']
    stat_char = ["m3"]

    parkingIDship = ["s1", "s2","s3","s4"]
    parkingIDship_1 = ["-E1_0", "E12_0", "-E12_0", "-E26_0"]
    shipNumber = [10,0,0,0]
    stat_charship = ["s1", "s2","s3","s4"]

    startparking(parkingID, motoNumber)
    startparkingship(parkingIDship, shipNumber)

    for i in range(len(new_list_ship)):
        ship_id = "ship_" + str(i)
        shipOBJ_dic[ship_id] = ship(ship_id, "s1")
    for i in range(len(new_list)):
        moto_id = "moto_" + str(i)
        if i <= motoNumber[0]-1:
            motOBJ_dic[moto_id] = moto(moto_id, "m1")
        if i >= motoNumber[0] and i <= motoNumber[1]+motoNumber[0]-1:
            motOBJ_dic[moto_id] = moto(moto_id, "m2")
    for i in range(len(parkingID)):
        prk_id = "m" + str(i+1)
        parkOBJ_dic[prk_id] = pconcent(prk_id)
    for i in range(len(stat_char)):
        stat_id = "m" + str(i+3)
        statOBJ_dic[stat_id] = stat_charg(stat_id)
    for i in range(len(stat_charship)):
        stat_id = "s" + str(i+1)
        stat_ship_OBJ_dic[stat_id] = stat_charg_ship(stat_id)
    for i in range(len(parkingIDship)):
        prkship_id = "s" + str(i+1)
        parkshipOBJ_dic[prkship_id] = pshipconcent(prkship_id)

    or_m = ['-114907924#3','-102276174#3','399732226#2','399732222#3','-114907908#4']
    des_m = ['-372051587#0','400299306#10','400299309#4','806220445#0','117572522#0']
    d_ship = [['s2'], []]

    while step < tsim:
    
        traci.simulationStep()
        moto_park_list = []
        ship_park_list = []
        destinationEdge_ship = []
        originEdge = []
        destinationEdge = []
        moto_list = traci.vehicle.getIDList()

        for j in parkingID:
            m_p_l = parkOBJ_dic[j].count(j)
            moto_park_list.extend(m_p_l[0])

        for j in parkingIDship:
            s_p_l = parkshipOBJ_dic[j].count(j)
            ship_park_list.extend(s_p_l[0])

        if step == 100:
            originEdge = [or_m[route-1]]
            destinationEdge = [des_m[route-1]]

        if step == 1000:
            if(route==1):
                destinationEdge_ship = d_ship[0]
            else:
                destinationEdge_ship = d_ship[1]
        if step < 100:
            for m in moto_list:
                if m[0:4] == 'ship':
                    t_ship = shipOBJ_dic[m].ship_state()
                    shipOBJ_dic[m].E_ship(m,t_ship,step)
                    shipOBJ_dic[m].vel_bus(m)

                if m[0:4] == 'moto':
                    t_s = motOBJ_dic[m].moto_state()[1]
                    if m == 'moto_1':
                        sssssssss=0
                    motOBJ_dic[m].E(m,t_s,step)
                    #if m == 'moto_0':
                        #print(t_s)

        if step >= 100:
            count_dest_ship = 0
            if destinationEdge_ship != []:
                shipOBJ_dic[ship_park_list[0]].v_spark_spark(destinationEdge_ship[0])
                count_dest_ship = 1


            if originEdge != []:
                for i in range(len(originEdge)):
                    if i ==9:
                        ddddd=0
                    #pop = random.choice(moto_park_list)
                    pop = moto_park_list[0]
                    motOBJ_dic[pop].viaje_OD(originEdge[i], destinationEdge[i])
                    m_l = pop
                    m_list.append(m_l)
                    m_1_list.append(m_l)
                    m_2_list.append(m_l)
                    moto_park_list.remove(pop)
            
            for m in moto_list:

                if m[0:4] == 'ship':
                    t_ship = shipOBJ_dic[m].ship_state()
                    shipOBJ_dic[m].E_ship(m,t_ship,step)
                    shipOBJ_dic[m].vel_bus(m)


                if m[0:4] == 'moto':
                    t_s = motOBJ_dic[m].moto_state()[1]
                    motOBJ_dic[m].E(m,t_s,step)
                    #if m == 'moto_0':
                    #    print(t_s)
                    #    print(step)

            if m_list != []:
                for m_1 in m_list:
                    if motOBJ_dic[m_1].moto_state()[1] == 3:
                        if motOBJ_dic[m_1].Energy[step] >= 3430:
                            motOBJ_dic[m_1].moto_peti(4)
                        elif motOBJ_dic[m_1].Energy[step] < 3430: #3691
                            motOBJ_dic[m_1].moto_peti(5)
                        m_1_list.remove(m_1)
                m_list = m_1_list

            motos_action = []
            motos_action_1 = []
            motos_action_3 = []
            ship_charge = []
            for m_1_2 in moto_list:
                if m_1_2[0:4] == 'moto':
                    if motOBJ_dic[m_1_2].trip_c == 4 or motOBJ_dic[m_1_2].trip_c == 5:
                        motos_action.append(m_1_2)
                    if motOBJ_dic[m_1_2].moto_state()[1] != 0:
                        motos_action_1.append(m_1_2)

            for m_2 in motos_action:
                if m_2[0:4] == 'moto':
                    if motOBJ_dic[m_2].trip_c == 4:
                        dist_moto =motOBJ_dic[m_2].m_dist(m_2,parkingID_1)
                        d_m = dist_moto.astype(int) 
                        sort = sorted(d_m)
                        pos_0= np.where(d_m == sort[0])
                        pos_1= np.where(d_m == sort[1])
                        cnt_1 = parkOBJ_dic[parkingID[int(pos_0[0])]].count(parkingID[int(pos_0[0])]) 
                        cnt_2 = parkOBJ_dic[parkingID[int(pos_1[0])]].count(parkingID[int(pos_1[0])])
                        cnt_t = list(cnt_1[0]) + list(cnt_2[0])
                        c_1 = list(cnt_1)[1]
                        c_2 = list(cnt_2)[1]
                        i_1 = 1
                        i_2 = 1

                        for m_3 in motos_action_1:
                            if (motOBJ_dic[m_3].moto_state()[1] == 1 or motOBJ_dic[m_3].moto_state()[1] == 2 or motOBJ_dic[m_3].moto_state()[1] == 3) and  (motOBJ_dic[m_3].Parking == 'm1' or  motOBJ_dic[m_3].Parking == 'm2'):
                                if motOBJ_dic[m_3].Parking == parkingID[int(pos_0[0])]:
                                    c_1 = c_1 - i_1
                                if motOBJ_dic[m_3].Parking == parkingID[int(pos_1[0])]:
                                    c_2= c_2 - i_2

                        if c_1 > 0:
                            motOBJ_dic[m_2].viaje_stat1(parkingID[int(pos_0[0])]) 
                            motOBJ_dic[m_2].moto_peti(0)
                        elif c_2 > 0:
                            motOBJ_dic[m_2].viaje_stat1(parkingID[int(pos_1[0])])  
                            motOBJ_dic[m_2].moto_peti(0)
                        else:  
                            motOBJ_dic[m_2].viaje_stat1(parkingID[int(pos_0[0])])
                            motOBJ_dic[m_2].moto_peti(6)

                    if motOBJ_dic[m_2].trip_c == 5:
                        j_1 = 1
                        dist_moto =motOBJ_dic[m_2].m_dist(m_2,stat_char_1)
                        d_m = dist_moto.astype(int) 
                        sort = sorted(d_m)
                        pos_0= np.where(d_m == sort[1])

                        if pos_0[0].shape[0] == 1: 
                            cntt_1 = statOBJ_dic[stat_char[int(pos_0[1])]].count(stat_char[int(pos_0[1])]) 
                            ct_1 = list(cntt_1)[1]
                            
                            for m_4 in motos_action_1:
                                if (motOBJ_dic[m_4].moto_state()[1] == 1 or motOBJ_dic[m_4].moto_state()[1] == 2 or motOBJ_dic[m_4].moto_state()[1] == 3) and  (motOBJ_dic[m_4].Parking == 'm3'):
                                    if motOBJ_dic[m_4].Parking == stat_char[int(pos_0[0])]:
                                        ct_1 = ct_1 - j_1
                                        
                            if ct_1 > 0:
                                motOBJ_dic[m_2].viaje_stat(stat_char[int(pos_0[0])]) 
                                motOBJ_dic[m_2].moto_peti(0)
                            else: 
                                motOBJ_dic[m_2].viaje_stat(stat_char[int(pos_0[0])]) 
                                motOBJ_dic[m_2].moto_peti(7)

                        else:
                            cntt_1 = statOBJ_dic[stat_char[int(pos_0[0][0])]].count(stat_char[int(pos_0[0][0])]) 
                            ct_1 = list(cntt_1)[1]

                            for m_4 in motos_action_1:
                                if (motOBJ_dic[m_4].moto_state()[1] == 1 or motOBJ_dic[m_4].moto_state()[1] == 2 or motOBJ_dic[m_4].moto_state()[1] == 3) and  (motOBJ_dic[m_4].Parking == 'm3'):
                                    if motOBJ_dic[m_4].Parking == stat_char[int(pos_0[0][0])]:
                                        ct_1 = ct_1 - j_1
                                        
                            if ct_1 > 0:
                                motOBJ_dic[m_2].viaje_stat(stat_char[int(pos_0[0][0])]) 
                                motOBJ_dic[m_2].moto_peti(0)
                            else: 
                                if step == 364:
                                    ssss=0
                                motOBJ_dic[m_2].viaje_stat(stat_char[int(pos_0[0][0])]) 
                                motOBJ_dic[m_2].moto_peti(7)


                    if motOBJ_dic[m_2].trip_c == 7:
                        a=0
            
            demanss=[]
            for m_dem in motos_action_1:
                if (motOBJ_dic[m_dem].Parking == 'm3') and (motOBJ_dic[m_dem].moto_state()[1] == 2 or motOBJ_dic[m_dem].moto_state()[1] == 1) and statOBJ_dic['m3'].count('m3')[1] > 0:
                    demanss.append(3700 - motOBJ_dic[m_dem].State_bater)
                    demanss_1.append(3700 - motOBJ_dic[m_dem].State_bater)

            for m_2_2_2 in moto_list:
                if m_2_2_2[0:4] == 'moto':
                    if (motOBJ_dic[m_2_2_2].Parking == 'm1' or motOBJ_dic[m_2_2_2].Parking == 'm2')  and motOBJ_dic[m_2_2_2].moto_state()[1] == 2:
                        motOBJ_dic[m_2_2_2].getin_stat1(motOBJ_dic[m_2_2_2].Parking)
            
            if step % 1000 == 0:
                Demandas[step] = sum(demanss)
        
            if step % 1000 == 0: # la simulación comienza a los 100 segundos, entonces intervalo de 500 y 500
                E_1 = list(tuple(reversed(traci.lane.getLastStepVehicleIDs ('-400299303#12_0'))) + tuple(reversed(traci.lane.getLastStepVehicleIDs ('400299303#12_0'))))
                dif = set(motos_action_1).difference(set(E_1))

                E_1 = list(tuple(reversed(traci.lane.getLastStepVehicleIDs ('-400299303#12_0'))) + tuple(dif)+ tuple(reversed(traci.lane.getLastStepVehicleIDs ('400299303#12_0'))))


                if len(E_1) >= 10:
                    for m_2_2 in E_1[0:10]:#motos_action_1[0:10]: #E_1:

                        #if (motOBJ_dic[m_2_2].Parking == 'm1' or motOBJ_dic[m_2_2].Parking == 'm2')  and motOBJ_dic[m_2_2].moto_state()[1] == 2:
                        #    motOBJ_dic[m_2_2].getin_stat1(motOBJ_dic[m_2_2].Parking)

                        if (motOBJ_dic[m_2_2].Parking == 'm3') and (motOBJ_dic[m_2_2].moto_state()[1] == 2 or motOBJ_dic[m_2_2].moto_state()[1] == 1) and statOBJ_dic['m3'].count('m3')[1] > 0: 
                            motOBJ_dic[m_2_2].getin_stat(motOBJ_dic[m_2_2].Parking)
                elif len(E_1) < 10:
                    for m_2_2 in E_1: #E_1:

                        #if (motOBJ_dic[m_2_2].Parking == 'm1' or motOBJ_dic[m_2_2].Parking == 'm2')  and motOBJ_dic[m_2_2].moto_state()[1] == 2:
                        #    motOBJ_dic[m_2_2].getin_stat1(motOBJ_dic[m_2_2].Parking)

                        if (motOBJ_dic[m_2_2].Parking == 'm3') and (motOBJ_dic[m_2_2].moto_state()[1] == 2 or motOBJ_dic[m_2_2].moto_state()[1] == 1) and statOBJ_dic['m3'].count('m3')[1] > 0: 
                            motOBJ_dic[m_2_2].getin_stat(motOBJ_dic[m_2_2].Parking)
                        
    # ------------------CARGA--------------------------------------------------------------------------------------------------------------------------
            motos_action_2 = []
            for m_1_3 in moto_list:
                if m_1_3[0:4] == 'moto':
                    if motOBJ_dic[m_1_3].moto_state()[1] == 0 and motOBJ_dic[m_1_3].Parking == 'm3':
                        motos_action_2.append(m_1_3)

            for m_1_4 in motos_action_2:
                statOBJ_dic[motOBJ_dic[m_1_4].Parking].charge(m_1_4,step)
    # ------------------CARGA--------------------------------------------------------------------------------------------------------------------------

    # ------------------VIAJE DE ESTACIÓN A PUNTO DE ACOPIO--------------------------------------------------------------------------------------------
            for m_1_5 in motos_action_2:
                if (motOBJ_dic[m_1_5].moto_state()[1] == 0 and motOBJ_dic[m_1_5].Parking == 'm3') and motOBJ_dic[m_1_5].Energy[step] == 3700:
                    St = motOBJ_dic[m_1_5].Parking
                    dist_moto_2 =motOBJ_dic[m_1_5].m_dist_park(parkingID_1,St)
                    d_m_1 = dist_moto_2.astype(int)
                    sort_1 = sorted(d_m_1)
                    pos_0_1= np.where(d_m_1 == sort_1[0])
                    pos_1_1= np.where(d_m_1 == sort_1[1])
                    cnt_1_1 = parkOBJ_dic[parkingID[int(pos_0_1[0])]].count(parkingID[int(pos_0_1[0])]) #cuenta los espacios disponibles en la estacion de carga
                    cnt_2_1 = parkOBJ_dic[parkingID[int(pos_1_1[0])]].count(parkingID[int(pos_1_1[0])]) 
                    cnt_t_0 = list(cnt_1_1[0]) + list(cnt_2_1[0])

                    c_1_1 = list(cnt_1_1)[1]
                    c_2_1 = list(cnt_2_1)[1]
                    i_1 = 1
                    i_2 = 1

                    for m_6 in motos_action_1:
                        if (motOBJ_dic[m_6].moto_state()[1] == 1 or motOBJ_dic[m_6].moto_state()[1] == 2 or motOBJ_dic[m_6].moto_state()[1] == 3) and  (motOBJ_dic[m_6].Parking == 'm1' or  motOBJ_dic[m_6].Parking == 'm2'):
                            if motOBJ_dic[m_6].Parking == parkingID[int(pos_0_1[0])]:
                                c_1_1 = c_1_1 - i_1
                            if motOBJ_dic[m_6].Parking == parkingID[int(pos_1_1[0])]:
                                c_2_1= c_2_1 - i_2

                    if c_1_1 > 0:
                        motOBJ_dic[m_1_5].v_stat_park(parkingID[int(pos_0_1[0])]) 
                        motOBJ_dic[m_1_5].moto_peti(0)
                    elif c_2_1 > 0:
                        motOBJ_dic[m_1_5].v_stat_park(parkingID[int(pos_1_1[0])])  
                        motOBJ_dic[m_1_5].moto_peti(0)
                    else:  
                        motOBJ_dic[m_1_5].v_stat_park(parkingID[int(pos_0_1[0])])
                        motOBJ_dic[m_1_5].moto_peti(6)
    # ------------------VIAJE DE ESTACIÓN A PUNTO DE ACOPIO--------------------------------------------------------------------------------------------

    # ------------------ EMBARCACIÓN-------------------------------------------------------------------------------------------------
            
            for m_s in moto_list:
                if m_s[0:4] == 'ship':
                    t_ship = shipOBJ_dic[m_s].ship_state()                

                    if shipOBJ_dic[m_s].Parkingship == 's2' and shipOBJ_dic[m_s].ship_state() == 0 and (m_s in traci.parkingarea.getVehicleIDs('s2')):

                        if shipOBJ_dic[m_s].S_E[step-1] < int(92*4*3.7*87/1.0212):
                            ch_ship = 1
                        if ch_ship == 1 and shipOBJ_dic[m_s].S_E[step] < int(92*4*3.7*87):
                            stat_ship_OBJ_dic['s2'].charge(m_s,step)
                            if shipOBJ_dic[m_s].S_E[step] == int(92*4*3.7*87):
                                ch_ship = 0
                            
                        if shipOBJ_dic[m_s].S_E[step-1] >= int(92*4*3.7*87/1.0212) and ch_ship == 0:
                            shipOBJ_dic[m_s].S_E[step] = shipOBJ_dic[m_s].S_E[step-1]
                            shipOBJ_dic[m_s].v_spark_spark('s3')

                    if shipOBJ_dic[m_s].Parkingship == 's3' and shipOBJ_dic[m_s].ship_state() == 0 and (m_s in traci.parkingarea.getVehicleIDs('s3')):

                        if shipOBJ_dic[m_s].S_E[step-1] < int(92*4*3.7*87/1.0212):
                            ch_ship = 2
                        if ch_ship == 2 and shipOBJ_dic[m_s].S_E[step] < int(92*4*3.7*87):
                            stat_ship_OBJ_dic['s3'].charge(m_s,step)
                            if shipOBJ_dic[m_s].S_E[step] == int(92*4*3.7*87):
                                ch_ship = 0

                        if shipOBJ_dic[m_s].S_E[step-1] >= int(92*4*3.7*87/1.0212) and ch_ship == 0:
                            shipOBJ_dic[m_s].S_E[step] = shipOBJ_dic[m_s].S_E[step-1]
                            shipOBJ_dic[m_s].v_spark_spark('s4')

                    if shipOBJ_dic[m_s].Parkingship == 's4' and shipOBJ_dic[m_s].ship_state() == 0 and (m_s in traci.parkingarea.getVehicleIDs('s4')):

                        if shipOBJ_dic[m_s].S_E[step-1] < int(92*4*3.7*87/1.0212):
                            ch_ship = 3
                        if ch_ship == 3 and shipOBJ_dic[m_s].S_E[step] < int(92*4*3.7*87):
                            stat_ship_OBJ_dic['s4'].charge(m_s,step)
                            if step == 317:
                                ssssss=0
                            if shipOBJ_dic[m_s].S_E[step] == int(92*4*3.7*87):
                                ch_ship = 0                    

                        if shipOBJ_dic[m_s].S_E[step-1] >= int(92*4*3.7*87/1.0212) and ch_ship == 0:
                            shipOBJ_dic[m_s].S_E[step] = shipOBJ_dic[m_s].S_E[step-1]
                            shipOBJ_dic[m_s].v_spark_spark('s1')
                            
                    if shipOBJ_dic[m_s].Parkingship == 's1' and shipOBJ_dic[m_s].ship_state() == 0 and (m_s in traci.parkingarea.getVehicleIDs('s1')):

                        if shipOBJ_dic[m_s].S_E[step-1] < int(92*4*3.7*87/1.0212):
                            ch_ship = 4
                        if ch_ship == 4 and shipOBJ_dic[m_s].S_E[step] < int(92*4*3.7*87):
                            stat_ship_OBJ_dic['s1'].charge(m_s,step)
                            if shipOBJ_dic[m_s].S_E[step] == int(92*4*3.7*87):
                                ch_ship = 0 

                        if shipOBJ_dic[m_s].S_E[step-1] >= int(92*4*3.7*87/1.0212) and ch_ship == 0:
                            shipOBJ_dic[m_s].S_E[step] = shipOBJ_dic[m_s].S_E[step-1]
        velocidad_ship.append(traci.vehicle.getSpeed('ship_0'))
        step += 1


    a=1
    operationGraphics()
    traci.close()

def operationGraphics():
        
    Demanda_a = np.zeros(25)
    Demanda_a[12] = demanss_1[0][0]
    r1,r2,r3,r4,r5,r6,r7,r8,tim, tim_2 = QP(Demanda_a)

    tmp = range(len(motOBJ_dic['moto_0'].Energy[0:2000]))
    tmp1 = range(len(motOBJ_dic['moto_0'].Energy[0:2000]))

    motOBJ_dic['moto_0'].Energy[0:100] = 3700
    motOBJ_dic['moto_0'].speed [0:100] = 0
    motOBJ_dic['moto_0'].Energy1[0:100] = 0
    motOBJ_dic['moto_0'].combus[0:100] = 0
    motOBJ_dic['moto_0'].combus2[0:100] = 0

    fig, (ax1) = plt.subplots(1)
    ax1.plot(tmp[0:2000], motOBJ_dic['moto_0'].speed[0:2000])
    ax1.set_title("Velocidad de la motocicleta")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Velocidad [m/s]")
    plt.savefig(STATIC_OPERATION_PATH+'fig1.png')

    fig, axs = plt.subplots(2, 1)
    axs[0].plot(tmp[0:2000], motOBJ_dic['moto_0'].Energy[0:2000])
    axs[0].set_title("Consumo energético de la motocicleta")
    axs[0].set_xlabel("tiempo [s]")
    axs[0].set_ylabel("Energía[Wh]")
    axs[1].plot(tmp[0:2000], motOBJ_dic['moto_0'].Energy1[0:2000])
    axs[1].set_title("Aporte eléctrico")
    axs[1].set_xlabel("Tiempo [s]")
    axs[1].set_ylabel("Potencia [Wh]")
    plt.tight_layout()
    plt.savefig(STATIC_OPERATION_PATH+'fig2.png')

    fig, (ax1,ax2) = plt.subplots(2)
    ax1.plot(tmp[0:2000], motOBJ_dic['moto_0'].combus[0:2000])
    ax1.set_title("Consumo de gasolina")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Gasolina [galones]")
    ax2.plot(tmp[0:2000], motOBJ_dic['moto_0'].combus2[0:2000])
    ax2.set_title("Aporte combustión")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Potencia [Wh]")
    plt.tight_layout()
    plt.savefig(STATIC_OPERATION_PATH+'fig3.png')



    tmp = range(len(shipOBJ_dic['ship_0'].S_E))

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(tmp, velocidad_ship)
    axs[0].set_title("Velocidad de la embarcación")
    axs[0].set_xlabel("Tiempo [s]")
    axs[0].set_ylabel("Velocidad [m/s]")
    axs[1].plot(tmp, shipOBJ_dic['ship_0'].S_E)
    axs[1].set_title("Consumo energético de la embarcación")
    axs[1].set_xlabel("tiempo [s]")
    axs[1].set_ylabel("Energía[Wh]")
    axs[2].plot(tmp, shipOBJ_dic['ship_0'].pot/3600)
    axs[2].set_title("Aporte eléctrico")
    axs[2].set_xlabel("Tiempo [s]")
    axs[2].set_ylabel("Potencia [W]")
    plt.tight_layout()
    plt.savefig(STATIC_OPERATION_PATH+'fig4.png')


    tmp = range(len(Demandas))

    fig, (ax1) = plt.subplots(1)
    ax1.plot(tmp, Demandas)
    ax1.set_title("Demandas")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Energía[Wh]")
    plt.savefig(STATIC_OPERATION_PATH+'fig5.png')

    fig, (ax1) = plt.subplots(1)
    ax1.plot(np.arange(0,len(r8),1), r8, "#ffd343", label = "Precio")
    ax1.set_title('Precio de compra')
    ax1.legend(loc = 'best', fontsize = 10)
    ax1.set_xlabel('Tiempo [h]')
    ax1.set_ylabel('Precio [USD]')
    plt.savefig(STATIC_OPERATION_PATH+'fig6.png')

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(tim_2, r2, "#ffd343", label = "Ganancia")
    ax1.set_title('Ganancia del operador de la estación de carga')
    ax1.legend(loc = 'best', fontsize = 10)
    ax1.set_xlabel('Tiempo [h]')
    ax1.set_ylabel('Ganancia [USD]')

    ax2.plot(tim_2, r7, "b", label = "Compra-venta-red")
    ax2.set_title('Balance energético')
    ax2.plot(tim_2, r4, "#ff7700", label = "Demanda") #E_f[:,0]
    ax2.plot(tim_2, r5, "r", label = "Almacenamiento")
    ax2.plot(tim_2, r6,"g" , label = "Fotovoltaica")
    ax2.legend(loc = 4, fontsize = 8)
    ax2.set_xlabel('Tiempo [h]')
    ax2.set_ylabel('Energía [W]')
    plt.tight_layout()
    plt.savefig(STATIC_OPERATION_PATH+'fig7.png')


    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(tim, r3, label = "Energía de la batería")
    ax1.set_title('Energía de la batería')
    ax1.legend(loc = 'best', fontsize = 10)
    ax1.set_xlabel('Tiempo [h]')
    ax1.set_ylabel(' Energía [Wh]')

    ax2.plot(tim_2, r1,'o', label = "Energía suministrada")
    ax2.set_title('Energía suministrada a la demanda')
    ax2.plot(tim_2, r4, "#ff7700", label = "Demanda") #E_f[:,0]
    ax2.legend(loc = 'best', fontsize = 10)
    ax2.set_xlabel('Tiempo [h]')
    ax2.set_ylabel('Demanda [W]')
    plt.tight_layout()
    plt.savefig(STATIC_OPERATION_PATH+'fig8.png')