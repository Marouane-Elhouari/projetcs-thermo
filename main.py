def calcul_pression_ideale(n, T, V):
    R = 8.314 
    P = (n * R * T) / V
    return P
n = 1.0
T = 300.0
V = 0.024

pression = calcul_pression_ideale(n, T, V)
print("Resultat",pression)