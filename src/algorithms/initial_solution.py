"""
Génération de solution initiale pour le VRP
Méthode: Nearest Neighbor avec time windows
"""

import numpy as np
from typing import List, Dict, Tuple
import sys

from src.models.solution import Solution, Route

class InitialSolutionGenerator:
    """Génère une solution initiale de qualité"""
    
    def __init__(self, cost_matrix: np.ndarray, 
                 demands: Dict[int, float],
                 time_windows: Dict[int, Tuple[float, float]],
                 vehicle_capacities: Dict[str, float],
                 config: Dict):
        
        self.cost_matrix = cost_matrix
        self.demands = demands
        self.time_windows = time_windows
        self.vehicle_capacities = vehicle_capacities
        self.config = config
        self.num_customers = cost_matrix.shape[0] - 1  # Exclure dépôt
        
    def generate_nearest_neighbor(self) -> Solution:
        """
        Génère solution initiale par plus proche voisin
        avec respect des contraintes de capacité et time windows
        """
        solution = Solution(num_vehicles=self.config.get('max_vehicles', 20))
        unassigned = set(range(1, self.num_customers + 1))
        vehicle_id = 0
        
        while unassigned and vehicle_id < solution.num_vehicles:
            # Créer nouvelle route
            vehicle_type = self._select_vehicle_type(unassigned)
            capacity = self.vehicle_capacities[vehicle_type]
            route = Route(vehicle_id, vehicle_type, capacity)
            route.customers = [0]  # Commencer au dépôt
            
            current_load = 0.0
            current_time = self.config.get('start_time', 6.0)  # 6h du matin
            current_node = 0
            
            while unassigned:
                # Trouver le client le plus proche réalisable
                next_customer = self._find_nearest_feasible(
                    current_node, unassigned, current_load, 
                    capacity, current_time, route
                )
                
                if next_customer is None:
                    break
                
                # Ajouter le client
                route.add_customer(next_customer)
                unassigned.remove(next_customer)
                
                # Mettre à jour charge et temps
                current_load += self.demands.get(next_customer, 0)
                travel_time = self._get_travel_time(current_node, next_customer)
                arrival_time = current_time + travel_time
                
                # Respecter time window
                tw_start, tw_end = self.time_windows.get(next_customer, (0, 24))
                if arrival_time < tw_start:
                    waiting_time = tw_start - arrival_time
                    arrival_time = tw_start
                else:
                    waiting_time = 0
                
                route.arrival_times[next_customer] = arrival_time
                route.waiting_times[next_customer] = waiting_time
                
                service_time = self.config.get('service_time', 0.5)  # 30 min
                current_time = arrival_time + service_time
                current_node = next_customer
                
            # Retour au dépôt
            route.add_customer(0)
            
            if not route.is_empty():
                solution.add_route(route)
                vehicle_id += 1
        
        # Clients non assignés
        solution.unassigned_customers = list(unassigned)
        
        # Calculer métriques
        solution.calculate_metrics(self.cost_matrix, self.demands, self.config)
        
        print(f"✓ Solution initiale: {len(solution.routes)} routes, "
              f"{len(unassigned)} clients non desservis")
        
        return solution
    
    def _find_nearest_feasible(self, current_node: int, 
                              unassigned: set, 
                              current_load: float,
                              capacity: float,
                              current_time: float,
                              route: Route) -> int:
        """Trouve le client le plus proche réalisable"""
        
        best_customer = None
        best_cost = float('inf')
        
        for customer in unassigned:
            # Vérifier capacité
            if current_load + self.demands.get(customer, 0) > capacity:
                continue
            
            # Vérifier time window
            travel_time = self._get_travel_time(current_node, customer)
            arrival_time = current_time + travel_time
            tw_start, tw_end = self.time_windows.get(customer, (0, 24))
            
            if arrival_time > tw_end + self.config.get('time_window_tolerance', 1.0):
                continue
            
            # Calculer coût
            cost = self.cost_matrix[current_node, customer]
            
            # Pénalité pour attente
            if arrival_time < tw_start:
                cost += (tw_start - arrival_time) * 100
            
            if cost < best_cost:
                best_cost = cost
                best_customer = customer
        
        return best_customer
    
    def _get_travel_time(self, from_node: int, to_node: int) -> float:
        """Calcule le temps de trajet (en heures)"""
        distance = self.cost_matrix[from_node, to_node]
        avg_speed = self.config.get('average_speed', 25)  # km/h
        return distance / avg_speed
    
    def _select_vehicle_type(self, remaining_customers: set) -> str:
        """Sélectionne le type de véhicule approprié"""
        # Logique simple: utiliser camionnette par défaut
        # Peut être amélioré selon les besoins
        total_demand = sum(self.demands.get(c, 0) for c in remaining_customers)
        
        if total_demand > 2000:
            return "Camion moyen"
        elif total_demand > 1000:
            return "Camionnette"
        else:
            return "Camionnette"
    
    def generate_savings(self) -> Solution:
        """
        Génère solution initiale par algorithme de Clarke-Wright Savings
        (Alternative plus sophistiquée)
        """
        # TODO: Implémenter Clarke-Wright si nécessaire
        pass