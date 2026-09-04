import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environnement_vrp import EnvironnementVRP


def masque_valide(env):
    return env.unwrapped.action_masks()


instances_test = [
    "RC101",
    "RC102",
    "RC103"
]

DOD = 0.3

CHEMIN_MODELE = "agent_vrp_v1"
CHEMIN_NORMALISATION = "vecnormalize_stats.pkl"


print("\n==========================================")
print(" DIAGNOSTIC DU MODELE PPO")
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

env_test = VecNormalize.load(
    CHEMIN_NORMALISATION,
    env_test
)

env_test.training = False
env_test.norm_reward = False


print("\nChargement du modèle...")

modele = MaskablePPO.load(
    CHEMIN_MODELE,
    env=env_test
)

print("Modèle chargé avec succès.")


observation = env_test.reset()

print("\n==========================================")
print(" ANALYSE DES ACTIONS")
print("==========================================")


for etape in range(10):

    action_mask = env_test.env_method(
        "action_masks"
    )[0]

    actions_valides = np.where(action_mask)[0]

    print("\n------------------------------------------")
    print(f"Étape {etape + 1}")
    print("------------------------------------------")

    print(
        "Position actuelle :",
        env_test.get_attr("position_actuelle")[0]
    )

    print(
        "Temps actuel :",
        env_test.get_attr("temps_actuel")[0]
    )

    print(
        "Capacité restante :",
        env_test.get_attr("capacite_restante")[0]
    )

    print(
        "Nombre d'actions valides :",
        len(actions_valides)
    )

    print(
        "Actions valides :",
        actions_valides.tolist()
    )

    action, _ = modele.predict(
        observation,
        action_masks=action_mask,
        deterministic=True
    )

    action = int(action[0])

    print(
        "Action choisie par PPO :",
        action
    )

    observation, recompense, termine, info = env_test.step(
        np.array([action])
    )

    if termine[0]:
        print("\nLa simulation est terminée.")
        break


print("\n==========================================")
print(" FIN DU DIAGNOSTIC")
print("==========================================")