"""
Étape 5 : Agent RL - Entraînement avec MaskablePPO
"""
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from environnement_vrp import EnvironnementVRP

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize
)

from stable_baselines3.common.monitor import Monitor


# ==========================================
# MASQUE DES ACTIONS
# ==========================================

def masque_valide(env):
    return env.unwrapped.action_masks()


# ==========================================
# ENTRAÎNEMENT
# ==========================================

if __name__ == "__main__":

    instances_entrainement = [
        "C101",
        "C102",
        "C103",
        "R101",
        "R102",
        "R103"
    ]

    instances_test = [
        "RC101",
        "RC102",
        "RC103"
    ]

    # ======================================
    # ENVIRONNEMENT
    # ======================================

    env = EnvironnementVRP(
        instances_entrainement,
        dod=0.3
    )

    # ======================================
    # WRAPPERS
    # ======================================

    env = Monitor(env)

    env = ActionMasker(
        env,
        masque_valide
    )

    env = DummyVecEnv(
        [lambda: env]
    )

    # ======================================
    # NORMALISATION
    # ======================================

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )

    # ======================================
    # PPO
    # ======================================

    modele = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1
    )

    # ======================================
    # ENTRAÎNEMENT
    # ======================================

    modele.learn(
        total_timesteps=50000
    )

    # ======================================
    # SAUVEGARDER PPO
    # ======================================

    modele.save(
        "agent_vrp_temps_reel"
    )

    # ======================================
    # SAUVEGARDER NORMALISATION
    # ======================================

    env.save(
        "vecnormalize_temps_reel.pkl"
    )

    # ======================================
    # SAUVEGARDER LE GNN
    # ======================================

    env_base = env.envs[0].unwrapped

    torch.save(
        env_base.gnn.state_dict(),
        "gnn_temps_reel.pth"
    )

    # ======================================
    # FIN
    # ======================================

    print("\n==========================================")
    print(" ENTRAÎNEMENT TERMINÉ")
    print("==========================================")

    print(
        "Modèle PPO sauvegardé sous : "
        "agent_vrp_temps_reel"
    )

    print(
        "Normalisation sauvegardée sous : "
        "vecnormalize_temps_reel.pkl"
    )

    print(
        "GNN sauvegardé sous : "
        "gnn_temps_reel.pth"
    )