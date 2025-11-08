"""
Configuration globale du projet VRP Yaounde
Contient tous les parametres du modele et de l'algorithme ALNS
"""

import os

# ============================================================
# CHEMINS DES FICHIERS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
GRAPHICS_DIR = os.path.join(RESULTS_DIR, 'graphiques')

# Fichier de donnees
DATA_FILE = os.path.join(DATA_DIR, 'trajets_yde.csv')

# ============================================================
# PARAMETRES DU MODELE VRP
# ============================================================

# Depot central (index 0 dans les solutions)
DEPOT_ID = 1  # ID 1 dans le CSV correspond au depot Mvan
DEPOT_NAME = "Dépôt Central (Mvan)"

# Capacites des vehicules (en unites de charge)
VEHICLE_CAPACITIES = {
    'Tous': 1000,           # Vehicule standard
    'Camionnette': 500,     # Petite capacite
    'Camion moyen': 1500,   # Moyenne capacite
    '4x4 recommandé': 800,  # 4x4 capacite moyenne
    '4x4': 800              # 4x4 obligatoire
}

# Couts fixes par type de vehicule (FCFA par jour)
VEHICLE_FIXED_COSTS = {
    'Tous': 10000,
    'Camionnette': 8000,
    'Camion moyen': 15000,
    '4x4 recommandé': 12000,
    '4x4': 12000
}

# Vitesses moyennes par type de route et periode (km/h)
SPEEDS = {
    'Bitumé': {
        'Faible': 45,        # Trafic faible
        'Moyen': 35,         # Trafic moyen
        'Dense': 20,         # Trafic dense
        'Très dense': 10     # Trafic tres dense
    },
    'Terre/Bitumé mixte': {
        'Faible': 30,
        'Moyen': 25,
        'Dense': 15,
        'Très dense': 8
    },
    'Terre/Latérite': {
        'Faible': 25,
        'Moyen': 20,
        'Dense': 12,
        'Très dense': 6
    }
}

# Temps de service moyen par client (minutes)
SERVICE_TIME = 15

# Facteurs de penalisation
PENALTY_LATE_DELIVERY = 5000    # FCFA par heure de retard
PENALTY_CAPACITY = 10000        # FCFA par unite de depassement
PENALTY_RESTRICTION = 20000     # FCFA si restriction violee
PENALTY_PARKING = 1000          # FCFA selon difficulte stationnement

# Poids de la fonction objectif multi-criteres
ALPHA_DISTANCE = 0.3   # Poids distance
BETA_COST = 0.4        # Poids cout monetaire
GAMMA_TIME = 0.2       # Poids temps
DELTA_VEHICLES = 0.1   # Poids nombre vehicules

# ============================================================
# PARAMETRES TEMPORELS
# ============================================================

# Fenetres temporelles par defaut (format 24h)
DEFAULT_TIME_WINDOWS = {
    'start': 6,    # 6h00
    'end': 18      # 18h00
}

# Periodes de la journee
PERIODS = {
    'matin': (6, 12),      # 6h-12h
    'midi': (12, 16),      # 12h-16h
    'soir': (16, 20)       # 16h-20h
}

# Heures de pointe a Yaounde
PEAK_HOURS_MORNING = (7, 10)   # 7h-10h
PEAK_HOURS_EVENING = (17, 20)  # 17h-20h

# ============================================================
# PARAMETRES ALGORITHME ALNS
# ============================================================

# Parametres generaux
MAX_ITERATIONS = 5000           # Nombre max d'iterations
SEGMENT_SIZE = 100              # Taille segment pour mise a jour poids
RANDOM_SEED = 42                # Graine aleatoire pour reproductibilite

# Simulated Annealing
INITIAL_TEMPERATURE = 1000.0    # Temperature initiale
COOLING_RATE = 0.995            # Taux de refroidissement (alpha)
MIN_TEMPERATURE = 1.0           # Temperature minimale

# Scores pour mise a jour des poids
SCORE_BEST_SOLUTION = 13        # Nouvelle meilleure solution globale
SCORE_BETTER_SOLUTION = 9       # Solution ameliorante
SCORE_ACCEPTED_SOLUTION = 3     # Solution acceptee (SA)
SCORE_REJECTED_SOLUTION = 0     # Solution rejetee

# Poids adaptatifs
LAMBDA_WEIGHT = 0.5             # Parametre lissage mise a jour poids (0-1)

# Destruction
Q_DESTROY_MIN = 0.1             # Pourcentage min clients a retirer
Q_DESTROY_MAX = 0.4             # Pourcentage max clients a retirer

# ============================================================
# PARAMETRES OPERATEURS
# ============================================================

# Noms des operateurs de destruction
DESTROY_OPERATORS = [
    'random_removal',
    'worst_removal',
    'shaw_removal',
    'time_based_removal',
    'route_removal'
]

# Noms des operateurs de reparation
REPAIR_OPERATORS = [
    'greedy_insertion',
    'regret_2_insertion',
    'regret_3_insertion',
    'time_oriented_insertion'
]

# Poids initiaux (uniformes)
INITIAL_DESTROY_WEIGHTS = [1.0] * len(DESTROY_OPERATORS)
INITIAL_REPAIR_WEIGHTS = [1.0] * len(REPAIR_OPERATORS)

# Parametres Shaw Removal
SHAW_ALPHA = 9.0    # Poids distance
SHAW_BETA = 3.0     # Poids temps
SHAW_GAMMA = 2.0    # Poids contraintes

# ============================================================
# CODIFICATION DES VARIABLES QUALITATIVES
# ============================================================

# Type de revetement
COATING_CODES = {
    'Bitumé': 3,
    'Terre/Bitumé mixte': 2,
    'Terre/Latérite': 1
}

# Etat de la route
ROAD_STATE_CODES = {
    'Bon': 3,
    'Moyen': 2,
    'Mauvais': 1
}

# Nids de poule
POTHOLE_CODES = {
    'Aucun': 4,
    'Peu': 3,
    'Quelques-uns': 2,
    'Nombreux': 1,
    'Très nombreux': 0
}

# Trafic
TRAFFIC_CODES = {
    'Faible': 4,
    'Moyen': 3,
    'Dense': 2,
    'Très dense': 1
}

# Securite
SECURITY_CODES = {
    'Élevée': 3,
    'Moyenne': 2,
    'Faible': 1
}

# Stationnement
PARKING_CODES = {
    'Excellent': 5,
    'Moyen': 4,
    'Difficile': 3,
    'Très difficile': 2,
    'Limité': 1,
    'Très limité': 0
}

# Restrictions horaires
RESTRICTION_CODES = {
    'Aucune': 4,
    'Jours matchs': 3,
    'Restrictions zone présidentielle': 2,
    '6h-10h seulement': 1,
    '6h-11h seulement': 1,
    'Éviter nuit': 1
}

# Type de vehicule requis
VEHICLE_TYPE_CODES = {
    'Tous': 5,
    'Camion moyen': 4,
    'Camionnette': 3,
    '4x4 recommandé': 2,
    '4x4': 1
}

# Fiabilite temps
RELIABILITY_CODES = {
    'Très élevée': 5,
    'Élevée': 4,
    'Moyenne': 3,
    'Faible': 2,
    'Très faible': 1
}

# Praticabilite pluie
RAIN_CODES = {
    'Excellente': 5,
    'Bonne': 4,
    'Moyenne': 3,
    'Mauvaise': 2,
    'Très mauvaise': 1
}

# ============================================================
# PARAMETRES VISUALISATION
# ============================================================

# Couleurs pour les graphiques
COLORS = {
    'route_1': '#FF6B6B',
    'route_2': '#4ECDC4',
    'route_3': '#45B7D1',
    'route_4': '#FFA07A',
    'route_5': '#98D8C8',
    'depot': '#2C3E50'
}

# Taille des figures
FIGURE_SIZE = (12, 8)

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'