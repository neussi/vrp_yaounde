"""
Module de gestion des solutions du VRP
Représente une solution (ensemble de routes) et ses métriques
"""

import copy
from typing import List, Dict, Tuple
import numpy as np

class Route:
    """Classe représentant une route individuelle"""
    
    def __init__(self, vehicle_id: int, vehicle_type: str, capacity: float):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.capacity = capacity
        self.customers = []  # Liste des clients [0, c1, c2, ..., cn, 0]
        self.load = 0.0
        self.total_distance = 0.0
        self.total_cost = 0.0
        self.total_time = 0.0
        self.arrival_times = {}  # {customer_id: arrival_time}
        self.waiting_times = {}  # {customer_id: waiting_time}
        
    def add_customer(self, customer_id: int, position: int = -1):
        """Ajoute un client à la route"""
        if position == -1:
            self.customers.append(customer_id)
        else:
            self.customers.insert(position, customer_id)
    
    def remove_customer(self, customer_id: int):
        """Retire un client de la route"""
        if customer_id in self.customers:
            self.customers.remove(customer_id)
            return True
        return False
    
    def is_empty(self) -> bool:
        """Vérifie si la route est vide (sauf dépôt)"""
        return len(self.customers) <= 2
    
    def get_customers_only(self) -> List[int]:
        """Retourne uniquement les clients (sans le dépôt)"""
        return [c for c in self.customers if c != 0]
    
    def __repr__(self):
        return f"Route {self.vehicle_id} ({self.vehicle_type}): {self.customers} | Load: {self.load:.1f}/{self.capacity} | Cost: {self.total_cost:.0f} FCFA"


class Solution:
    """Classe représentant une solution complète du VRP"""
    
    def __init__(self, num_vehicles: int = 10):
        self.routes = []
        self.num_vehicles = num_vehicles
        self.total_cost = 0.0
        self.total_distance = 0.0
        self.total_time = 0.0
        self.num_vehicles_used = 0
        self.unassigned_customers = []
        self.is_feasible = True
        self.constraint_violations = []
        
    def add_route(self, route: Route):
        """Ajoute une route à la solution"""
        self.routes.append(route)
        
    def remove_empty_routes(self):
        """Supprime les routes vides"""
        self.routes = [r for r in self.routes if not r.is_empty()]
        
    def get_all_customers(self) -> List[int]:
        """Retourne tous les clients desservis"""
        customers = []
        for route in self.routes:
            customers.extend(route.get_customers_only())
        return customers
    
    def calculate_metrics(self, cost_matrix: np.ndarray,
                         demands: Dict[int, float],
                         config: Dict,
                         distance_matrix: np.ndarray = None):
        """Calcule toutes les métriques de la solution"""
        self.total_cost = 0.0
        self.total_distance = 0.0
        self.total_time = 0.0
        self.num_vehicles_used = len([r for r in self.routes if not r.is_empty()])

        # Si pas de matrice de distances, utiliser cost_matrix (rétrocompatibilité)
        if distance_matrix is None:
            distance_matrix = cost_matrix

        for route in self.routes:
            if route.is_empty():
                continue

            # Calculer charge
            route.load = sum(demands.get(c, 0) for c in route.get_customers_only())

            # Calculer distance et coût réels
            route_cost = 0.0
            route_distance = 0.0

            for i in range(len(route.customers) - 1):
                from_node = route.customers[i]
                to_node = route.customers[i + 1]

                # Distance en km
                route_distance += distance_matrix[from_node, to_node]
                # Coût en FCFA
                route_cost += cost_matrix[from_node, to_node]

            # Ajouter coût fixe du véhicule
            route_cost += config.get('fixed_vehicle_cost', 5000)

            # Ajouter pénalités de surcharge si dépassement de capacité
            if route.load > route.capacity:
                overload = route.load - route.capacity
                overload_penalty = config.get('overload_penalty', 5000.0)
                route_cost += overload * overload_penalty

            route.total_distance = route_distance
            route.total_cost = route_cost

            self.total_distance += route_distance
            self.total_cost += route.total_cost

        # Pénalités pour clients non assignés
        self.total_cost += len(self.unassigned_customers) * config.get('unassigned_penalty', 50000)
        
    def copy(self):
        """Crée une copie profonde de la solution"""
        return copy.deepcopy(self)
    
    def __repr__(self):
        return (f"Solution: {self.num_vehicles_used} véhicules | "
                f"Coût total: {self.total_cost:.0f} FCFA | "
                f"Distance: {self.total_distance:.1f} km | "
                f"Clients non desservis: {len(self.unassigned_customers)}")