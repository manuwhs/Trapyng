#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:09:07 2019

@author: montoya
"""

import os
import sys
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
os.chdir("../../")
base_path = os.path.abspath('')
sys.path.append(base_path + "/")
import import_folders

# Own graphical library
from graph_lib import gl

gl.close("all")
import numpy as np


Total_votos = 100
Blue_party_prop = 0.7

Blue_votos = Blue_party_prop * Total_votos
Red_votos = Total_votos - Blue_votos
N_escanos = 9

### Divide between 

def get_thresholds_Group(total_parties, N_escanos):
    """
    Returns the thresholds of a given party. 
    You can give several at the same time using a list.
    """
    
    array_of_thresholds = [] 
    
    for distribution in total_parties:
        array_of_thresholds_i = []
        for i in range(N_escanos):
            for party_votes in distribution:
                array_of_thresholds_i.append(party_votes/(i+1))
            
        array_of_thresholds.append(np.array(array_of_thresholds_i))

    return array_of_thresholds

def get_esanhos_only_one_party(array_of_thresholds,threshold_corte_ultimo_escano_list):
    party_escanhos = np.zeros((threshold_corte_ultimo_escano_list.size, len(array_of_thresholds)))# List of (N_party_configurations, N_thresholds)
    
    for i in range(threshold_corte_ultimo_escano_list.size):
        threshold = threshold_corte_ultimo_escano_list[i]
        for j  in range(len(array_of_thresholds)):
            party_thresholds = array_of_thresholds[j]
            num_escanos_obtenidos = np.where(party_thresholds >= threshold )[0].size
            
            if (num_escanos_obtenidos > N_escanos):
                num_escanos_obtenidos = N_escanos
                
            party_escanhos[i,j] = num_escanos_obtenidos
    return party_escanhos

import random
def get_esanhos_between_two_formations(blue_group,red_group, threshold_corte_ultimo_escano_list,N_escanos):
    array_of_thresholds_blue = get_thresholds_Group ([blue_group],N_escanos )
    array_of_thresholds_red = get_thresholds_Group ([red_group],N_escanos )
    
    party_escanhos_blue = np.zeros((threshold_corte_ultimo_escano_list.size, len(array_of_thresholds_blue)))# List of (N_party_configurations, N_thresholds)
    party_escanhos_red = np.zeros((threshold_corte_ultimo_escano_list.size, len(array_of_thresholds_red)))# List of (N_party_configurations, N_thresholds)
        
    for i in range(threshold_corte_ultimo_escano_list.size):
        threshold = threshold_corte_ultimo_escano_list[i]
#        print (threshold)
        num_escanos_blue = np.where(array_of_thresholds_blue>= threshold )[0].size
        num_escanos_red = np.where(array_of_thresholds_red>= threshold )[0].size

        if (num_escanos_blue + num_escanos_red >= N_escanos):
            
            if (num_escanos_blue + num_escanos_red > N_escanos):
            ## We distribute the last one at random.
                ## Obtain rand integer between the number of parties
                winner_party = random.randint(1, len(blue_group) + len(red_group))
                
                # Resolver empate !!
                if (winner_party > len(blue_group)): # Wins red
                    num_escanos_blue -= 1
                else:
                    num_escanos_red -= 1
            party_escanhos_blue[i,0] = num_escanos_blue
            party_escanhos_red[i,0] = num_escanos_red
            break
            
        party_escanhos_blue[i,0] = num_escanos_blue
        party_escanhos_red[i,0] = num_escanos_red
        
    for j in range(i+1,threshold_corte_ultimo_escano_list.size):
        party_escanhos_blue[j,0] = party_escanhos_blue[j-1,0] 
        party_escanhos_red[j,0] = party_escanhos_red[j-1,0] 

    return party_escanhos_blue, party_escanhos_red

"""
Generate distribution of votes blue and red

"""

## Blue
total_parties_blue = [[Blue_votos], [Blue_votos/2,Blue_votos/2] , [Blue_votos/3,Blue_votos/3, Blue_votos/3]]
array_of_thresholds = get_thresholds_Group (total_parties_blue,N_escanos )
threshold_corte_ultimo_escano_list_blue = np.linspace(0,Total_votos,10000)
party_escanhos_blue = get_esanhos_only_one_party (array_of_thresholds, threshold_corte_ultimo_escano_list_blue)

## Red
total_parties_red = [[Red_votos], [Red_votos/2,Red_votos/2] , [Red_votos/3,Red_votos/3, Red_votos/3]]
array_of_thresholds = get_thresholds_Group (total_parties_red,N_escanos )
threshold_corte_ultimo_escano_list_red = np.linspace(0,Total_votos,10000)
party_escanhos_red = get_esanhos_only_one_party (array_of_thresholds, threshold_corte_ultimo_escano_list_red)


## Comparison
threshold_corte_ultimo_escano_list = np.linspace(Total_votos,0,10000)

blue_party = total_parties_blue[2]
red_party = total_parties_red[1]
party_escanhos_blue_2, party_escanhos_red_2 = get_esanhos_between_two_formations(blue_party,red_party, threshold_corte_ultimo_escano_list,N_escanos)
 

"""
PLOT THE INDIVIDUAL ESCANHOS AND THE TOTAL IN THE END
"""


gl.init_figure();
ax1 = gl.subplot2grid((3,1), (0,0), rowspan=1, colspan=1)
ax2 = gl.subplot2grid((3,1), (1,0), rowspan=1, colspan=1, sharex = ax1, sharey = ax1)
ax3 = gl.subplot2grid((3,1), (2,0), rowspan=1, colspan=1, sharex = ax1, sharey = ax1)
    
gl.plot(threshold_corte_ultimo_escano_list_blue,party_escanhos_blue, ax = ax1,
        labels = ["Numero de escanhos obtenidos en funcion del numero de votos del ultimo escanho", "Numero de votos del ultimo escanho", "Numero de escanhos obtenidos"],
        legend = [["%.2f"%(total_parties_blue[i][j]) for j in range(len(total_parties_blue[i]))] for i in range(len(total_parties_blue))])

gl.plot(threshold_corte_ultimo_escano_list_red,party_escanhos_red, ax = ax2,
        labels = ["Numero de escanhos obtenidos en funcion del numero de votos del ultimo escanho", "Numero de votos del ultimo escanho", "Numero de escanhos obtenidos"],
        legend = [["%.2f"%(total_parties_red[i][j]) for j in range(len(total_parties_red[i]))] for i in range(len(total_parties_red))])


gl.plot(threshold_corte_ultimo_escano_list,party_escanhos_blue_2, ax = ax3,
        labels = ["", "", ""],
        legend = ["Blue "+ str(blue_party)], color = "b")

gl.plot(threshold_corte_ultimo_escano_list,party_escanhos_red_2, ax = ax3,
        labels = ["", "", ""],
        legend = ["Red "+ str(red_party)], color = "r")

"""
CONCLUSION: DADO UN THRESHOLD, EL NUMERO DE VOTOS DEL ULTIMO ESCANHO DADO, LA MEJOR CONFIGURACION POSIBLE ES TENERLO TODO REPARTIDO.

OTRA INFORMACION: EL THRESHOLD FINAL CAMBIA EN FUNCION DE LA COMPETENCIA. 

PREGUNTA: ES POSIBLE QUE SEA MEJOR TENER UNA DISTRIBUCION MAS REPARTIDA DEL VOTO PARA QUE EL THRESHOLD SE SITUE UN POCO MAYOR 


"""