import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environnement_vrp import EnvironnementVRP


# =========================================================
# Fonction pour récupérer le masque des actions valides
# =========================================================

def masque_valide(env):
    return env.unwrapped.action_masks()


# =========================================================
# PARAMÈTRES
# =========================================================

instances_test = [
    "RC101",
    "RC102",
    "RC103"
]

DOD = 0.3

CHEMIN_MODELE = "agent_vrp_v1"
CHEMIN_NORMALISATION = "vecnormalize_stats.pkl"


# =========================================================
# 1. Création de l'environnement
# =========================================================

print("\n==========================================")
print(" SIMULATION DU TEMPS RÉEL")
print("==========================================")

env_test = EnvironnementVRP(
    instances_disponibles=instances_test,
    dod=DOD
)

env_test = Monitor(env_test)

env_test = ActionMasker(
    env_test,
    masque_valide
)

env_test = DummyVecEnv([
    lambda: env_test
])


# =========================================================
# 2. Charger la normalisation utilisée pendant
#    l'entraînement
# =========================================================

env_test = VecNormalize.load(
    CHEMIN_NORMALISATION,
    env_test
)

# Très important :
# pendant le test, on ne modifie plus les statistiques
# de normalisation apprises pendant l'entraînement.

env_test.training = False

# On veut observer les vraies récompenses
env_test.norm_reward = False


# =========================================================
# 3. Charger l'agent PPO entraîné
# =========================================================

print("\nChargement du modèle PPO...")

modele = MaskablePPO.load(
    CHEMIN_MODELE,
    env=env_test
)

print("Modèle chargé avec succès.")


# =========================================================
# 4. Démarrer une simulation
# =========================================================

observation = env_test.reset()

termine = False
tronque = False

recompense_totale = 0.0
compteur_etapes = 0

print("\n------------------------------------------")
print("Début de la tournée")
print("------------------------------------------")


# =========================================================
# 5. Boucle de simulation
# =========================================================

while not termine and not tronque and compteur_etapes < 500:

    # Récupérer le masque des actions actuellement valides
    action_mask = env_test.env_method(
        "action_masks"
    )[0]

    # Afficher les clients actuellement disponibles
    clients_disponibles = np.where(action_mask)[0]

    print("\n==========================================")
    print(f"Étape : {compteur_etapes + 1}")

    print(
        f"Temps actuel : "
        f"{env_test.get_attr('temps_actuel')[0]:.2f}"
    )

    print(
        f"Position actuelle : "
        f"{env_test.get_attr('position_actuelle')[0]}"
    )

    print(
        f"Capacité restante : "
        f"{env_test.get_attr('capacite_restante')[0]:.2f}"
    )

    print(
        f"Clients disponibles : "
        f"{clients_disponibles.tolist()}"
    )


    # =====================================================
    # Décision de l'agent
    # =====================================================

    action, _ = modele.predict(
        observation,
        action_masks=action_mask,
        deterministic=True
    )

    action = int(action[0])

    print(f"Action choisie par PPO : client {action}")


    # =====================================================
    # Exécuter l'action
    # =====================================================

    observation, recompense, termine_array, info = env_test.step(
        np.array([action])
    )

    termine = bool(termine_array[0])

    recompense_etape = float(recompense[0])

    recompense_totale += recompense_etape

    print(
        f"Récompense de l'étape : "
        f"{recompense_etape:.2f}"
    )

    compteur_etapes += 1


# =========================================================
# 6. Résultat final
# =========================================================

print("\n\n==========================================")
print(" FIN DE LA SIMULATION")
print("==========================================")

print(
    f"Nombre d'étapes : {compteur_etapes}"
)

print(
    f"Récompense totale : {recompense_totale:.2f}"
)

print(
    f"Temps final : "
    f"{env_test.get_attr('temps_actuel')[0]:.2f}"
)

print(
    f"Distance / déplacement final : "
    f"position {env_test.get_attr('position_actuelle')[0]}"
)

if termine:
    print("\nLa tournée est terminée.")
else:
    print("\nLa simulation s'est arrêtée avant la fin.")

