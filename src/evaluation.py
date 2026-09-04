import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environnement_vrp import EnvironnementVRP


def masque_valide(env):
    return env.unwrapped.action_masks()


# ==========================================
# PARAMÈTRES
# ==========================================

instances_test = [
    "RC101",
    "RC102",
    "RC103"
]

DOD = 0.3

CHEMIN_MODELE = "agent_vrp_temps_reel"
CHEMIN_NORMALISATION = "vecnormalize_temps_reel.pkl"


# ==========================================
# CRÉATION DE L'ENVIRONNEMENT
# ==========================================

env = EnvironnementVRP(
    instances_disponibles=instances_test,
    dod=DOD
)

env = Monitor(env)

env = ActionMasker(
    env,
    masque_valide
)

env = DummyVecEnv([
    lambda: env
])

env = VecNormalize.load(
    CHEMIN_NORMALISATION,
    env
)

env.training = False
env.norm_reward = False


# ==========================================
# CHARGEMENT DU MODÈLE
# ==========================================

print("\n==========================================")
print(" ÉVALUATION DU MODÈLE PPO")
print("==========================================")

print("\nChargement du modèle...")

modele = MaskablePPO.load(
    CHEMIN_MODELE,
    env=env
)

print("Modèle chargé avec succès.")


# ==========================================
# INITIALISATION
# ==========================================

observation = env.reset()

termine = False
tronque = False

recompense_totale = 0.0
distance_totale = 0.0
temps_total = 0.0


nombre_retours_depot = 0
nombre_clients_visites = 0
nombre_etapes = 0

clients_visites = set()


# ==========================================
# SIMULATION
# ==========================================

while not termine and not tronque and nombre_etapes < 500:

    # Position avant l'action
    position_avant = env.get_attr(
        "position_actuelle"
    )[0]

    # Masque des actions valides
    action_mask = env.env_method(
        "action_masks"
    )[0]

    # Action choisie par PPO
    action, _ = modele.predict(
        observation,
        action_masks=action_mask,
        deterministic=True
    )

    action = int(action[0])

    # --------------------------------------
    # Calcul de la distance
    # --------------------------------------

    if action != position_avant:

        distance = env.get_attr(
            "matrice_distances"
        )[0][position_avant][action]

        distance_totale += distance

    # --------------------------------------
    # Comptage des clients
    # --------------------------------------

    if action != 0:
        clients_visites.add(action)

    # Retour au dépôt
    if action == 0 and position_avant != 0:
        nombre_retours_depot += 1


    temps_avant = env.get_attr(
      "temps_actuel"
    )[0]

    # --------------------------------------
    # Exécution de l'action
    # --------------------------------------

    observation, recompense, termine_array, info = env.step(
        np.array([action])
    )


    temps_apres = env.get_attr("temps_actuel")[0]


    termine = bool(termine_array[0])

    if termine:
       temps_total = temps_avant + distance
    else:
       temps_total = temps_apres


    recompense_etape = float(recompense[0])

    recompense_totale += recompense_etape

    nombre_etapes += 1


# ==========================================
# RÉSULTATS
# ==========================================




nombre_clients_visites = len(clients_visites)

nombre_clients_total = env.get_attr(
    "n_clients"
)[0] - 1

clients_non_visites = (
    nombre_clients_total - nombre_clients_visites
)




# ==========================================
# AFFICHAGE
# ==========================================

print("\n==========================================")
print(" RÉSULTATS DE L'ÉVALUATION")
print("==========================================")

print(
    f"\nInstances possibles : "
    f"{instances_test}"
)

print(
    f"DoD : "
    f"{DOD}"
)

print(
    f"Nombre total de clients : "
    f"{nombre_clients_total}"
)

print(
    f"Clients visités : "
    f"{nombre_clients_visites}"
)

print(
    f"Clients non visités : "
    f"{clients_non_visites}"
)

print(
    f"Nombre de retours au dépôt : "
    f"{nombre_retours_depot}"
)

print(
    f"Nombre d'étapes : "
    f"{nombre_etapes}"
)

print(
    f"Distance totale parcourue : "
    f"{distance_totale:.2f}"
)

print(
    f"Temps total : "
    f"{temps_total:.2f}"
)

print(
    f"Récompense totale : "
    f"{recompense_totale:.2f}"
)

print("\n==========================================")

if clients_non_visites == 0:
    print("La tournée est complète.")
else:
    print(
        f"La tournée n'est pas complète : "
        f"{clients_non_visites} client(s) non visité(s)."
    )