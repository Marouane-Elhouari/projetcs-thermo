import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

def uniquac_binary(x1, T, r1, r2, q1, q2, a12, a21):
    """
    Calcule les coefficients d'activité avec le modèle UNIQUAC pour un mélange binaire.
    
    Bloc 1 : Initialisation et Paramètres Fixes
    ------------------------------------------
    R : Constante universelle des gaz parfaits [J/mol.K]
    z : Nombre de coordination (fixé à 10 pour le modèle standard)
    """
    R = 8.314 
    z = 10 
    x2 = 1 - x1

    # Bloc 2 : Fractions de Volume (Phi) et de Surface (Theta)
    # -------------------------------------------------------
    # Ces paramètres traduisent la taille (r) et la forme (q) des molécules.
    phi1 = x1 * r1 / (x1*r1 + x2*r2)
    phi2 = x2 * r2 / (x1*r1 + x2*r2)
    
    theta1 = x1 * q1 / (x1*q1 + x2*q2)
    theta2 = x2 * q2 / (x1*q1 + x2*q2)

    # Bloc 3 : Contribution Combinatoire (Entropie)
    # --------------------------------------------
    # Calcul de l'écart à l'idéalité dû aux différences de géométrie moléculaire.
    l1 = (z / 2) * (r1 - q1) - (r1 - 1)
    l2 = (z / 2) * (r2 - q2) - (r2 - 1)

    ln_gamma1_comb = np.log(phi1/x1) + (z/2)*q1*np.log(theta1/phi1) + \
                     l1 - (phi1/x1)*(x1*l1 + x2*l2)
    ln_gamma2_comb = np.log(phi2/x2) + (z/2)*q2*np.log(theta2/phi2) + \
                     l2 - (phi2/x2)*(x1*l1 + x2*l2)

    # Bloc 4 : Contribution Résiduelle (Enthalpie / Interactions)
    # ----------------------------------------------------------
    # Calcul de l'énergie d'interaction entre les différentes espèces chimiques.
    tau12 = np.exp(-a12 / T)
    tau21 = np.exp(-a21 / T)

    ln_gamma1_res = q1 * (1 - np.log(theta1 + theta2*tau21) - \
                    theta1/(theta1 + theta2*tau21) - \
                    theta2*tau12/(theta2 + theta1*tau12))
    
    ln_gamma2_res = q2 * (1 - np.log(theta2 + theta1*tau12) - \
                    theta2/(theta2 + theta1*tau12) - \
                    theta1*tau21/(theta1 + theta2*tau21))

    # Bloc 5 : Assemblage et Conversion
    # ---------------------------------
    # Somme des contributions et passage à l'exponentielle pour obtenir gamma.
    gamma1 = np.exp(ln_gamma1_comb + ln_gamma1_res)
    gamma2 = np.exp(ln_gamma2_comb + ln_gamma2_res)

    return gamma1, gamma2

# --- SECTION EXEMPLE : ACÉTONE (1) - CHLOROFORME (2) ---
T = 323.15  # 50°C en Kelvin

# Paramètres r (volume) et q (surface)
r1, q1 = 2.574, 2.336  # Acétone
r2, q2 = 2.870, 2.410  # Chloroforme

# Paramètres d'interaction binaire (K)
a12, a21 = -52.39, 340.35

# Simulation sur toute la plage de composition
x1_range = np.linspace(0.001, 0.999, 100)
g1, g2 = uniquac_binary(x1_range, T, r1, r2, q1, q2, a12, a21)

# Bloc 6 : Validation Thermodynamique (Gibbs-Duhem)
# ------------------------------------------------
# L'aire sous les courbes de ln(gamma) doit satisfaire la cohérence thermodynamique.
area1 = simpson(np.log(g1) * x1_range, x1_range)
area2 = simpson(np.log(g2) * (1 - x1_range), x1_range)

print(f"Différence Gibbs-Duhem : {abs(area1 - area2):.6f} (doit être proche de 0)")

# Bloc 7 : Visualisation Graphique
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(x1_range, g1, 'b-', label='Acétone ($\gamma_1$)')
plt.plot(x1_range, g2, 'r-', label='Chloroforme ($\gamma_2$)')
plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
plt.xlabel('Fraction molaire $x_{acetone}$')
plt.ylabel('Coefficient d\'activité $\gamma$')
plt.title('Modèle UNIQUAC : Mélange Acétone-Chloroforme')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()