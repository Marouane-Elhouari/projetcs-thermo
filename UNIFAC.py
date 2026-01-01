import numpy as np
import matplotlib.pyplot as plt


class MoleculeUNIFAC:
    """Classe pour définir une molécule en groupes UNIFAC"""
    """
    Au lieu de traiter chaque molécule individuellement,
    UNIFAC décompose les molécules en groupes fonctionnels.
    """

    def __init__(self, name, groups):
        """
        Paramètres :
        -----------
        name : str
            Nom de la molécule
        groups : dict
            Dictionnaire {nom_groupe : nombre}
        """
        self.name = name
        self.groups = groups

        """
        mol = MoleculeUNIFAC("Éthanol", {"CH3": 1, "CH2": 1, "OH": 1})

                 mol.name = "Éthanol"
                 mol.groups = {"CH3": 1, "CH2": 1, "OH": 1}
        """

        # Paramètres de groupes (exemples)
              # Dictionnaire R : Volumes de Van der Waals
        self.R = {
            'CH3': 0.9011,
            'CH2': 0.6744,
            'CH': 0.4469,
            'OH': 1.0000,
            'H2O': 0.9200,
            'CH3CO': 1.6724
        }
              # Dictionnaire Q : Surfaces de Van der Waals 

        self.Q = {
            'CH3': 0.848,
            'CH2': 0.540,
            'CH': 0.228,
            'OH': 1.200,
            'H2O': 1.400,
            'CH3CO': 1.488
        }

        # Calcul des paramètres moléculaires
        """ 
        {"CH3": 1, "CH2": 1, "OH": 1}.items()
        for grp, n in groups.items() → boucle qui extrait chaque paire

               Itération 1 : grp = "CH3", n = 1
               Itération 2 : grp = "CH2", n = 1
               Itération 3 : grp = "OH",  n = 1


        n * self.R[grp] → multiplication nombre x volume

        Itération 1 : 1 x 0.9011 = 0.9011
        Itération 2 : 1 x 0.6744 = 0.6744
        Itération 3 : 1 x 1.0000 = 1.0000
        """
        self.r = sum(n * self.R[grp] for grp, n in groups.items())
        self.q = sum(n * self.Q[grp] for grp, n in groups.items())
    
    """   ethanol = MoleculeUNIFAC("Ethanol", {"CH3": 1, "CH2": 1, "OH": 1})
                       print(ethanol)
                       Sortie : Molecule (Ethanol) : r = 2.5755, q = 2.5880, groupes = {'CH3': 1, 'CH2': 1, 'OH': 1}     
                 """
    def __repr__(self):
        return ( 
            f"Molecule ({self.name}) : "
            f"r = {self.r:.4f}, q = {self.q:.4f}, "
            f"groupes = {self.groups}"
        )


def unifac_combinatorial(x1, mol1, mol2):
    """
    Calcul de la partie combinatorielle UNIFAC

    Paramètres :
    -----------
    x1 : float ou array
        Fraction molaire du constituant 1
    mol1, mol2 : MoleculeUNIFAC
        Objets molécules

    Retourne :
    ---------
    ln_gamma1_comb, ln_gamma2_comb : float ou array
    """
    x2 = 1 - x1
    z = 10  # Nombre de coordination

    r1, r2 = mol1.r, mol2.r
    q1, q2 = mol1.q, mol2.q

    # Fractions de volume et de surface
    phi1 = x1 * r1 / (x1 * r1 + x2 * r2)
    phi2 = x2 * r2 / (x1 * r1 + x2 * r2)

    theta1 = x1 * q1 / (x1 * q1 + x2 * q2)
    theta2 = x2 * q2 / (x1 * q1 + x2 * q2)

    # Paramètres l
    l1 = (z / 2) * (r1 - q1) - (r1 - 1)
    l2 = (z / 2) * (r2 - q2) - (r2 - 1)

    # Partie combinatorielle
    ln_gamma1_comb = (
        np.log(phi1 / x1)
        + (z / 2) * q1 * np.log(theta1 / phi1)
        + l1
        - (phi1 / x1) * (x1 * l1 + x2 * l2)
    )

    ln_gamma2_comb = (
        np.log(phi2 / x2)
        + (z / 2) * q2 * np.log(theta2 / phi2)
        + l2
        - (phi2 / x2) * (x1 * l1 + x2 * l2)
    )

    return ln_gamma1_comb, ln_gamma2_comb


# ==========================
# Définition des molécules
# ==========================

# Éthanol : CH3–CH2–OH
ethanol = MoleculeUNIFAC("Ethanol", {"CH3": 1, "CH2": 1, "OH": 1})

# Eau : H2O
eau = MoleculeUNIFAC("Eau", {"H2O": 1})

# Acétone : CH3–CO–CH3
acetone = MoleculeUNIFAC("Acetone", {"CH3": 2, "CH3CO": 1})

print("Paramètres moléculaires UNIFAC :")
print("=" * 50)
print(ethanol)
print(eau)
print(acetone)
print()

# ==========================
# Calcul Éthanol (1) – Eau (2)
# ==========================

"""
Cette fonction crée un tableau de valeurs espacées uniformément entre deux bornes.
       Syntaxe : np.linspace(début, fin, nombre_de_points)
"""
x1_range = np.linspace(0.01, 0.99, 99)

"""
Au lieu de faire une boucle :
ln_gamma1_comb = []
ln_gamma2_comb = []
for x1 in x1_range:
    g1, g2 = unifac_combinatorial(x1_range, ethanol, eau)
    gamma1_comb.append(g1)
    gamma2_comb.append(g2)

NumPy le fait automatiquement en une ligne.
"""
ln_gamma1_comb, ln_gamma2_comb = unifac_combinatorial(
    x1_range, ethanol, eau
)

gamma1_comb = np.exp(ln_gamma1_comb)
gamma2_comb = np.exp(ln_gamma2_comb)

# Affichage des résultats
print("Contribution combinatorielle UNIFAC – Éthanol (1) / Eau (2)")
print("=" * 60)
print(f"{'x_ethanol':<12} {'gamma1_comb':<15} {'gamma2_comb':<15}")
print("-" * 60)

for i in [0, 24, 49, 74, 98]:
    print(
        f"{x1_range[i]:<12.3f} "
        f"{gamma1_comb[i]:<15.4f} "
        f"{gamma2_comb[i]:<15.4f}"
    )

# ==========================
# Graphique
# ==========================

    # Définit les étiquettes des axes et le titre.
plt.figure(figsize=(10, 6))
plt.plot(x1_range, gamma1_comb, 'b-', linewidth=2, label="Éthanol (comb.)")
plt.plot(x1_range, gamma2_comb, 'r-', linewidth=2, label="Eau (comb.)")
plt.axhline(y=1, linestyle='--', alpha=0.3)
plt.xlabel("Fraction molaire éthanol, x", fontsize=12)
plt.ylabel("Coefficient d’activité (partie combinatorielle)", fontsize=12)
plt.title("UNIFAC – Contribution combinatorielle Éthanol / Eau",
          fontsize=13, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("unifac_combinatorial.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nNote : Le calcul complet UNIFAC nécessite les paramètres")
print("d’interaction entre groupes (disponibles dans les tables DECHEMA).")

