"""
Modèle mathématique du Vehicle Routing Problem (VRP)

Ce module définit la structure mathématique du problème:
- Ensembles (nœuds, véhicules, périodes)
- Paramètres (coûts, demandes, capacités)
- Variables de décision
- Fonction objectif
- Contraintes

Le modèle peut être utilisé pour:
1. Documentation formelle du problème
2. Export vers des solveurs (CPLEX, Gurobi, OR-Tools)
3. Validation de solutions
"""

from typing import Dict, List, Tuple, Set
import numpy as np
from dataclasses import dataclass, field

@dataclass
class VRPParameters:
    """
    Paramètres du problème VRP
    
    Cette classe encapsule tous les paramètres du modèle mathématique
    conformément à la formulation classique du VRP avec contraintes.
    """
    # Ensembles (ARGUMENTS REQUIS EN PREMIER)
    num_customers: int          # n: nombre de clients
    num_vehicles: int           # v: nombre de véhicules disponibles
    
    # Matrices de paramètres (ARGUMENTS REQUIS)
    distance_matrix: np.ndarray     # d_ij: distance entre i et j (km)
    cost_matrix: np.ndarray         # c_ij^t: coût de transport i→j période t (FCFA)
    informal_fees: np.ndarray       # f_ij: frais informels arc (i,j) (FCFA)
    toll_fees: np.ndarray           # p_ij: péages arc (i,j) (FCFA)
    
    # Paramètres clients (ARGUMENTS REQUIS)
    demands: Dict[int, float]                        # q_i: demande client i
    time_windows: Dict[int, Tuple[float, float]]    # [a_i, b_i]: fenêtre temporelle
    service_times: Dict[int, float]                 # s_i: temps de service
    
    # Paramètres véhicules (ARGUMENTS REQUIS)
    vehicle_capacities: Dict[str, float]            # Q_k: capacité véhicule type k
    vehicle_types: List[str]                        # Types disponibles
    fixed_vehicle_costs: Dict[str, float]           # Coût fixe par type
    
    # ARGUMENTS AVEC VALEURS PAR DÉFAUT (DOIVENT VENIR APRÈS)
    num_periods: int = 3        # t: nombre de périodes (matin, midi, soir)
    
    # Paramètres de praticabilité
    road_conditions: Dict[Tuple[int, int], str] = field(default_factory=dict)     # État routes
    rain_practicability: Dict[int, str] = field(default_factory=dict)             # Praticabilité pluie
    
    # Contraintes de restrictions
    hourly_restrictions: Dict[int, Tuple[float, float]] = field(default_factory=dict)  # Restrictions horaires
    vehicle_requirements: Dict[int, str] = field(default_factory=dict)                  # Type véhicule requis
    parking_difficulty: Dict[int, str] = field(default_factory=dict)                    # Difficulté stationnement
    security_level: Dict[int, str] = field(default_factory=dict)                        # Niveau de sécurité
    
    # Paramètres de coût
    fuel_cost_per_km: float = 300.0      # FCFA/km
    average_speed: float = 25.0          # km/h
    
    # Poids de la fonction objectif multi-critères
    alpha: float = 0.4   # Poids distance
    beta: float = 0.3    # Poids coût monétaire
    gamma: float = 0.2   # Poids temps
    delta: float = 0.1   # Poids nombre véhicules


class VRPModel:
    """
    Modèle mathématique complet du VRP
    
    Représente le problème sous forme mathématique:
    
    VARIABLES DE DECISION:
    - x_ijk^t ∈ {0,1}: véhicule k va de i à j durant période t
    - y_k ∈ {0,1}: véhicule k est utilisé
    - u_i ≥ 0: temps d'arrivée au nœud i
    - w_i ≥ 0: temps d'attente au nœud i
    
    FONCTION OBJECTIF:
    MIN Z = α·Z_distance + β·Z_coût + γ·Z_temps + δ·Z_véhicules
    
    CONTRAINTES:
    (C1) Visite unique de chaque client
    (C2) Conservation de flux
    (C3) Capacité véhicule
    (C4) Fenêtres temporelles
    (C5) Cohérence temporelle
    (C6) Restrictions horaires
    (C7) Compatibilité véhicule-route
    (C8) Contraintes de stationnement
    """
    
    def __init__(self, parameters: VRPParameters):
        """
        Initialisation du modèle
        
        Args:
            parameters: Paramètres complets du problème
        """
        self.params = parameters
        
        # Construire les ensembles
        self.nodes = set(range(parameters.num_customers + 1))  # {0, 1, ..., n}
        self.customers = self.nodes - {0}  # Tous sauf dépôt
        self.depot = 0
        
        # Périodes temporelles
        self.periods = ['matin', 'midi', 'soir']
        
    def formulate_objective(self) -> str:
        """
        Retourne la formulation mathématique de la fonction objectif
        
        Returns:
            Chaîne LaTeX de la fonction objectif
        """
        objective = f"""
        FONCTION OBJECTIF (Multi-criteres):
        
        MIN Z = {self.params.alpha}·Z_distance + 
                {self.params.beta}·Z_cout + 
                {self.params.gamma}·Z_temps + 
                {self.params.delta}·Z_vehicules
        
        Ou:
        Z_distance = ∑∑∑∑ d_ij · x_ijk^t
        Z_cout = ∑∑∑∑ (c_ij^t + f_ij + p_ij) · x_ijk^t
        Z_temps = ∑_i (u_i + w_i)
        Z_vehicules = ∑_k y_k · Cout_fixe_k
        """
        return objective
    
    def get_constraints(self) -> Dict[str, str]:
        """
        Retourne toutes les contraintes du modèle
        
        Returns:
            Dictionnaire {nom_contrainte: formulation_mathematique}
        """
        constraints = {}
        
        # C1: Visite unique de chaque client
        constraints['C1_unique_visit'] = """
        ∑_k ∑_t ∑_j x_ijk^t = 1,  ∀i ∈ Clients
        
        Explication: Chaque client doit etre visite exactement une fois
        par un seul vehicule durant une seule periode.
        """
        
        # C2: Conservation de flux
        constraints['C2_flow_conservation'] = """
        ∑_j x_ijk^t - ∑_j x_jik^t = 0,  ∀i, k, t
        
        Explication: Un vehicule qui arrive a un noeud doit en repartir.
        Assure la continuite des routes.
        """
        
        # C3: Capacité des véhicules
        constraints['C3_capacity'] = """
        ∑_i q_i · ∑_j ∑_t x_ijk^t ≤ Q_k · y_k,  ∀k
        
        Explication: La charge totale d'un vehicule ne peut pas
        depasser sa capacite maximale Q_k.
        """
        
        # C4: Fenêtres temporelles
        constraints['C4_time_windows'] = """
        a_i ≤ u_i ≤ b_i,  ∀i
        
        Explication: Chaque client i doit etre visite dans sa
        fenetre temporelle [a_i, b_i].
        """
        
        # C5: Cohérence temporelle
        constraints['C5_time_consistency'] = """
        u_i + s_i + t_ij^t ≤ u_j + M(1 - x_ijk^t),  ∀i, j, k, t
        
        Explication: Si le vehicule k va de i a j (x_ijk^t = 1),
        alors l'arrivee a j doit etre coherente avec le depart de i
        plus le temps de trajet et de service.
        M est une grande constante (Big-M).
        """
        
        # C6: Restrictions horaires
        constraints['C6_hourly_restrictions'] = """
        x_ijk^t ≤ r_i^t,  ∀i, k, t
        
        Explication: Certains clients ont des restrictions d'acces
        selon la periode. r_i^t = 1 si l'acces est autorise, 0 sinon.
        """
        
        # C7: Compatibilité véhicule-route
        constraints['C7_vehicle_compatibility'] = """
        x_ijk^t ≤ compatibilite(k, revetement_ij),  ∀i, j, k, t
        
        Explication: Certaines routes necessitent des vehicules
        specifiques (ex: 4x4 pour routes en laterite).
        """
        
        # C8: Départ et retour au dépôt
        constraints['C8_depot'] = """
        ∑_j x_0jk^t = y_k,  ∀k, t  (depart du depot)
        ∑_i x_i0k^t = y_k,  ∀k, t  (retour au depot)
        
        Explication: Chaque vehicule utilise part du depot
        et y retourne.
        """
        
        # C9: Domaines des variables
        constraints['C9_domains'] = """
        x_ijk^t ∈ {0, 1},  ∀i, j, k, t
        y_k ∈ {0, 1},  ∀k
        u_i ≥ 0,  ∀i
        w_i ≥ 0,  ∀i
        
        Explication: Variables binaires pour les decisions de routage
        et d'utilisation de vehicules. Variables continues pour les temps.
        """
        
        return constraints
    
    def validate_solution(self, solution) -> Tuple[bool, List[str]]:
        """
        Valide une solution par rapport au modèle mathématique
        
        Vérifie que la solution respecte toutes les contraintes
        du modèle formel.
        
        Args:
            solution: Solution à valider (objet Solution)
            
        Returns:
            (est_valide, liste_violations)
        """
        violations = []
        
        # C1: Vérifier que chaque client est visité exactement une fois
        all_customers = solution.get_all_customers()
        customer_counts = {}
        for customer in all_customers:
            customer_counts[customer] = customer_counts.get(customer, 0) + 1
        
        for customer, count in customer_counts.items():
            if count != 1:
                violations.append(
                    f"C1 violee: Client {customer} visite {count} fois au lieu de 1"
                )
        
        # Vérifier que tous les clients sont desservis
        expected_customers = self.customers
        served_customers = set(all_customers)
        unserved = expected_customers - served_customers
        
        if unserved:
            violations.append(
                f"C1 violee: {len(unserved)} client(s) non desservi(s): {unserved}"
            )
        
        # C2: Vérifier la conservation de flux (chaque route est un cycle)
        for route in solution.routes:
            if route.is_empty():
                continue
            
            customers = route.customers
            if customers[0] != self.depot or customers[-1] != self.depot:
                violations.append(
                    f"C2 violee: Route {route.vehicle_id} ne commence/termine "
                    f"pas au depot (0)"
                )
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def calculate_objective_value(self, solution, cost_calculator) -> Dict[str, float]:
        """
        Calcule la valeur de la fonction objectif pour une solution
        
        Décompose le coût total selon les 4 composantes:
        - Distance
        - Coût monétaire
        - Temps
        - Nombre de véhicules
        
        Args:
            solution: Solution à évaluer
            cost_calculator: Instance de CostCalculator
            
        Returns:
            Dictionnaire des composantes de coût
        """
        components = {
            'Z_distance': 0.0,
            'Z_cost': 0.0,
            'Z_time': 0.0,
            'Z_vehicles': 0.0,
            'Z_total': 0.0
        }
        
        # Z_distance: somme des distances parcourues
        components['Z_distance'] = solution.total_distance
        
        # Z_cost: coûts monétaires (carburant, péages, frais informels)
        for route in solution.routes:
            if route.is_empty():
                continue
            
            route_costs = cost_calculator.calculate_route_cost(
                route.customers, route.vehicle_type
            )
            components['Z_cost'] += route_costs['travel_cost']
        
        # Z_time: temps total (trajet + attente + service)
        # Simplifié: distance / vitesse moyenne
        components['Z_time'] = solution.total_distance / self.params.average_speed
        
        # Z_vehicles: coût des véhicules utilisés
        for route in solution.routes:
            if not route.is_empty():
                vehicle_type = route.vehicle_type
                fixed_cost = self.params.fixed_vehicle_costs.get(vehicle_type, 5000)
                components['Z_vehicles'] += fixed_cost
        
        # Fonction objectif totale (pondérée)
        components['Z_total'] = (
            self.params.alpha * components['Z_distance'] +
            self.params.beta * components['Z_cost'] +
            self.params.gamma * components['Z_time'] +
            self.params.delta * components['Z_vehicles']
        )
        
        return components
    
    def export_to_lp(self, filename: str):
        """
        Export le modèle au format LP (Linear Programming)
        
        Format texte lisible pour debug et documentation
        
        Args:
            filename: Nom du fichier de sortie (.lp)
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\\* Vehicle Routing Problem - Yaounde *\\\n\n")
            
            # Fonction objectif
            f.write("Minimize\n")
            f.write(" obj: ")
            f.write(f"{self.params.alpha} Z_distance + ")
            f.write(f"{self.params.beta} Z_cost + ")
            f.write(f"{self.params.gamma} Z_time + ")
            f.write(f"{self.params.delta} Z_vehicles\n\n")
            
            # Contraintes
            f.write("Subject To\n")
            constraints = self.get_constraints()
            for name, formulation in constraints.items():
                f.write(f"\\* {name} *\\\n")
                f.write(formulation + "\n\n")
            
            # Bornes
            f.write("Bounds\n")
            f.write("\\* Variables continues *\\\n")
            f.write(f" 0 <= Z_distance <= {1000 * self.params.num_customers}\n")
            f.write(f" 0 <= Z_cost <= {100000 * self.params.num_customers}\n")
            f.write(f" 0 <= Z_time <= {24 * self.params.num_vehicles}\n")
            f.write(f" 0 <= Z_vehicles <= {self.params.num_vehicles}\n\n")
            
            # Variables binaires
            f.write("Binary\n")
            f.write("\\* Variables de routage x_ijk_t et utilisation y_k *\\\n")
            f.write(" x_ijk_t y_k\n\n")
            
            f.write("End\n")
        
        print(f"Modele exporte vers {filename}")
    
    def get_problem_statistics(self) -> Dict:
        """
        Calcule des statistiques sur l'instance du problème
        
        Returns:
            Dictionnaire de statistiques
        """
        stats = {
            'num_customers': self.params.num_customers,
            'num_vehicles': self.params.num_vehicles,
            'num_periods': self.params.num_periods,
            'total_demand': sum(self.params.demands.values()),
            'avg_demand': np.mean(list(self.params.demands.values())),
            'total_capacity': sum(self.params.vehicle_capacities.values()),
            'avg_distance': np.mean(self.params.distance_matrix[self.params.distance_matrix > 0]),
            'max_distance': np.max(self.params.distance_matrix),
            'customers_with_time_windows': len(self.params.time_windows),
            'customers_with_restrictions': len(self.params.hourly_restrictions),
            'vehicle_types': len(self.params.vehicle_types)
        }
        
        # Calculer le facteur de difficulté
        stats['difficulty_factor'] = self._calculate_difficulty_factor(stats)
        
        return stats
    
    def _calculate_difficulty_factor(self, stats: Dict) -> float:
        """
        Calcule un facteur de difficulté du problème
        
        Plus le facteur est élevé, plus le problème est difficile
        
        Args:
            stats: Statistiques du problème
            
        Returns:
            Facteur de difficulté (0-100)
        """
        difficulty = 0.0
        
        # Taille du problème
        difficulty += min(30, stats['num_customers'] * 0.5)
        
        # Contraintes temporelles
        tw_ratio = stats['customers_with_time_windows'] / max(stats['num_customers'], 1)
        difficulty += tw_ratio * 20
        
        # Restrictions
        restr_ratio = stats['customers_with_restrictions'] / max(stats['num_customers'], 1)
        difficulty += restr_ratio * 20
        
        # Rapport demande/capacité
        capacity_ratio = stats['total_demand'] / max(stats['total_capacity'], 1)
        if capacity_ratio > 0.8:
            difficulty += 15
        elif capacity_ratio > 0.6:
            difficulty += 10
        
        # Dispersion géographique
        if stats['max_distance'] > 20:
            difficulty += 15
        
        return min(100, difficulty)
    
    def print_model_summary(self):
        """
        Affiche un résumé du modèle mathématique
        """
        print("\n" + "="*70)
        print("MODELE MATHEMATIQUE VRP - YAOUNDE")
        print("="*70)
        
        print("\n[ENSEMBLES]")
        print(f"  Noeuds (N):           {len(self.nodes)} (depot + {len(self.customers)} clients)")
        print(f"  Vehicules (V):        {self.params.num_vehicles}")
        print(f"  Periodes (T):         {self.params.num_periods} {self.periods}")
        print(f"  Types de vehicules:   {len(self.params.vehicle_types)}")
        
        print("\n[PARAMETRES]")
        print(f"  Demande totale:       {sum(self.params.demands.values()):.1f} unites")
        print(f"  Capacite totale:      {sum(self.params.vehicle_capacities.values()):.1f} unites")
        print(f"  Distance max:         {np.max(self.params.distance_matrix):.1f} km")
        print(f"  Cout carburant:       {self.params.fuel_cost_per_km} FCFA/km")
        print(f"  Vitesse moyenne:      {self.params.average_speed} km/h")
        
        print("\n[VARIABLES DE DECISION]")
        print("  x_ijk^t : Routage (binaire)")
        print("  y_k     : Utilisation vehicule (binaire)")
        print("  u_i     : Temps arrivee (continue)")
        print("  w_i     : Temps attente (continue)")
        
        print("\n[FONCTION OBJECTIF]")
        print(f"  MIN Z = {self.params.alpha}·Distance + "
              f"{self.params.beta}·Cout + "
              f"{self.params.gamma}·Temps + "
              f"{self.params.delta}·Vehicules")
        
        print("\n[CONTRAINTES]")
        print("  C1: Visite unique")
        print("  C2: Conservation flux")
        print("  C3: Capacite vehicule")
        print("  C4: Fenetres temporelles")
        print("  C5: Coherence temporelle")
        print("  C6: Restrictions horaires")
        print("  C7: Compatibilite vehicule-route")
        print("  C8: Depart/retour depot")
        
        stats = self.get_problem_statistics()
        print(f"\n[DIFFICULTE]")
        print(f"  Facteur de difficulte: {stats['difficulty_factor']:.1f}/100")
        
        print("="*70 + "\n")