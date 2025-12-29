import numpy as np
import matplotlib.pyplot as plt

# On importe NumPy pour les calculs mathématiques (exponentielles, ln)
# et Matplotlib pour générer les graphiques des coefficients d'activité

def nrtl_binary(x1, T, a12, a21, alpha):
    """
    On définit cette fonction afin de la réutiliser pour le calcul des 
    coefficients d'activité de différents mélanges avec les paramètres de 
    l'équation NRTL : x, T, et les paramètres d'interaction a12 et a21.
    """
    # La fraction molaire totale est égale à 1 (100%)
    x2 = 1 - x1
    
    # Calcul des tau (Paramètres d'interaction binaire dépendants de T)
    tau12 = a12 / T
    tau21 = a21 / T
    
    # Calcul des G (Paramètres de non-randomness / probabilité locale)
    G12 = np.exp(-alpha * tau12)
    G21 = np.exp(-alpha * tau21)
    
    # Calcul des coefficients d'activité pour le constituant 1
    # On décompose le terme ln(gamma) en 2 parties (term1 et term2) pour plus de clarté
    term1_gamma1 = tau21 * (G21 / (x1 + x2 * G21))**2
    term2_gamma1 = tau12 * G12 / (x2 + x1 * G12)**2
    ln_gamma1 = x2**2 * (term1_gamma1 + term2_gamma1)
    
    # Calcul des coefficients d'activité pour le constituant 2
    term1_gamma2 = tau12 * (G12 / (x2 + x1 * G12))**2
    term2_gamma2 = tau21 * G21 / (x1 + x2 * G21)**2
    ln_gamma2 = x1**2 * (term1_gamma2 + term2_gamma2)
    
    # On transforme le ln(gamma) en gamma par la fonction exponentielle
    gamma1 = np.exp(ln_gamma1)
    gamma2 = np.exp(ln_gamma2)
    
    return gamma1, gamma2

# --- Application : Mélange Éthanol (1) - Eau (2) ---
T = 298.15      # Température en Kelvin (25°C)
a12 = -0.8009   # Paramètre d'interaction 1-2 (K)
a21 = 1239.5    # Paramètre d'interaction 2-1 (K)
alpha = 0.3     # Paramètre de non-randomness

# Génération d'une plage de compositions (x1 varie de 0.001 à 0.999)
x1_range = np.linspace(0.001, 0.999, 100)
gamma1_vals, gamma2_vals = nrtl_binary(x1_range, T, a12, a21, alpha)

# Note : Une fois que l'on exécute ces lignes, c'est comme si l'on avait 
# résolu l'équation complexe du modèle NRTL 100 fois d'un seul coup,
# allant de l'éthanol très dilué jusqu'à l'éthanol presque pur.

# --- Affichage des résultats (Points clés) ---
print("Résultats NRTL : Éthanol (1) - Eau (2) à 25°C")
print("-" * 50)
print(f"{'x_ethanol':<12} {'gamma_ethanol':<15} {'gamma_eau':<15}")
indices_a_afficher = [0, 24, 49, 74, 99]
for i in indices_a_afficher:
    print(f"{x1_range[i]:<12.3f} {gamma1_vals[i]:<15.4f} {gamma2_vals[i]:<15.4f}")

# --- Calcul à dilution infinie ---
# Pour l'éthanol : x1 = 0.001 (mélange à 99,9% d'eau). L'éthanol est "perdu" dans l'eau.
# Pour l'eau : x1 = 0.999 (mélange à 99,9% d'éthanol). L'eau est diluée à l'infini.
# gamma_inf ~ 4.45 signifie une forte répulsion des molécules d'éthanol par l'eau.

gamma1_inf = nrtl_binary(0.001, T, a12, a21, alpha)[0]
gamma2_inf = nrtl_binary(0.999, T, a12, a21, alpha)[1]

print("\nCoefficients à dilution infinie :")
print(f"gamma_1^inf (éthanol dans eau) = {gamma1_inf:.4f}")
print(f"gamma_2^inf (eau dans éthanol) = {gamma2_inf:.4f}")

# --- Tracé du Graphique ---

plt.figure(figsize=(8, 5))
plt.plot(x1_range, gamma1_vals, 'b-', label='Éthanol (1)')
plt.plot(x1_range, gamma2_vals, 'r-', label='Eau (2)')
plt.axhline(y=1, color='black', linestyle='--', alpha=0.3) # Ligne de l'idéalité (Raoult)
plt.xlabel('Fraction molaire éthanol (x1)')
plt.ylabel("Coefficient d'activité (gamma)")
plt.title("Coefficients d'activité NRTL - Éthanol/Eau")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()