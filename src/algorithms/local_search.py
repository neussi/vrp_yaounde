"""
Opérateurs de recherche locale pour amélioration de solution
2-opt, 3-opt, relocate, exchange
"""

import numpy as np
from typing import List
from src.models.solution import Solution
from typing import Dict, Tuple
from src.models.solution import Route

class LocalSearch:
    """Opérateurs de recherche locale"""
    
    def __init__(self, cost_matrix: np.ndarray, config: Dict):
        self.cost_matrix = cost_matrix
        self.config = config
    
    def two_opt(self, solution: Solution) -> Solution:
        """
        Amélioration 2-opt intra-route
        Inverse un segment de route pour réduire les croisements
        """
        improved = True
        new_solution = solution.copy()
        
        while improved:
            improved = False
            
            for route in new_solution.routes:
                if len(route.customers) <= 3:  # Trop petit pour 2-opt
                    continue
                
                for i in range(1, len(route.customers) - 2):
                    for j in range(i + 1, len(route.customers) - 1):
                        # Calculer gain
                        gain = self._calculate_2opt_gain(route, i, j)
                        
                        if gain > 0:
                            # Appliquer 2-opt
                            route.customers[i:j+1] = reversed(route.customers[i:j+1])
                            improved = True
                            break
                    
                    if improved:
                        break
        
        new_solution.calculate_metrics(self.cost_matrix, {}, self.config)
        return new_solution
    
    def relocate(self, solution: Solution) -> Solution:
        """
        Déplace un client d'une route à une autre
        """
        new_solution = solution.copy()
        improved = True
        
        while improved:
            improved = False
            
            for route1_idx, route1 in enumerate(new_solution.routes):
                for customer_idx in range(1, len(route1.customers) - 1):
                    customer = route1.customers[customer_idx]
                    
                    # Essayer de déplacer vers toutes les autres routes
                    for route2_idx, route2 in enumerate(new_solution.routes):
                        if route1_idx == route2_idx:
                            continue
                        
                        # Essayer toutes les positions dans route2
                        for insert_pos in range(1, len(route2.customers)):
                            gain = self._calculate_relocate_gain(
                                route1, customer_idx,
                                route2, insert_pos
                            )
                            
                            if gain > 0:
                                # Appliquer relocation
                                route1.customers.pop(customer_idx)
                                route2.customers.insert(insert_pos, customer)
                                improved = True
                                break
                        
                        if improved:
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        new_solution.remove_empty_routes()
        new_solution.calculate_metrics(self.cost_matrix, {}, self.config)
        return new_solution
    
    def exchange(self, solution: Solution) -> Solution:
        """
        Échange deux clients entre deux routes différentes
        """
        new_solution = solution.copy()
        improved = True
        
        while improved:
            improved = False
            
            for route1_idx, route1 in enumerate(new_solution.routes):
                for idx1 in range(1, len(route1.customers) - 1):
                    customer1 = route1.customers[idx1]
                    
                    for route2_idx, route2 in enumerate(new_solution.routes):
                        if route1_idx >= route2_idx:
                            continue
                        
                        for idx2 in range(1, len(route2.customers) - 1):
                            customer2 = route2.customers[idx2]
                            
                            gain = self._calculate_exchange_gain(
                                route1, idx1,
                                route2, idx2
                            )
                            
                            if gain > 0:
                                # Appliquer exchange
                                route1.customers[idx1] = customer2
                                route2.customers[idx2] = customer1
                                improved = True
                                break
                        
                        if improved:
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break
        
        new_solution.calculate_metrics(self.cost_matrix, {}, self.config)
        return new_solution
    
    def _calculate_2opt_gain(self, route: Route, i: int, j: int) -> float:
        """Calcule le gain d'un mouvement 2-opt"""
        customers = route.customers
        
        # Coût avant
        cost_before = (
            self.cost_matrix[customers[i-1], customers[i]] +
            self.cost_matrix[customers[j], customers[j+1]]
        )
        
        # Coût après
        cost_after = (
            self.cost_matrix[customers[i-1], customers[j]] +
            self.cost_matrix[customers[i], customers[j+1]]
        )
        
        return cost_before - cost_after
    
    def _calculate_relocate_gain(self, route1: Route, idx1: int,
                                 route2: Route, idx2: int) -> float:
        """Calcule le gain d'un mouvement de relocation"""
        customer = route1.customers[idx1]
        
        # Coût de retrait de route1
        cost_remove = (
            self.cost_matrix[route1.customers[idx1-1], customer] +
            self.cost_matrix[customer, route1.customers[idx1+1]] -
            self.cost_matrix[route1.customers[idx1-1], route1.customers[idx1+1]]
        )
        
        # Coût d'insertion dans route2
        cost_insert = (
            self.cost_matrix[route2.customers[idx2-1], customer] +
            self.cost_matrix[customer, route2.customers[idx2]] -
            self.cost_matrix[route2.customers[idx2-1], route2.customers[idx2]]
        )
        
        return cost_remove - cost_insert
    
    def _calculate_exchange_gain(self, route1: Route, idx1: int,
                                 route2: Route, idx2: int) -> float:
        """Calcule le gain d'un échange"""
        c1 = route1.customers[idx1]
        c2 = route2.customers[idx2]
        
        # Coût avant dans route1
        cost_r1_before = (
            self.cost_matrix[route1.customers[idx1-1], c1] +
            self.cost_matrix[c1, route1.customers[idx1+1]]
        )
        
        # Coût après dans route1
        cost_r1_after = (
            self.cost_matrix[route1.customers[idx1-1], c2] +
            self.cost_matrix[c2, route1.customers[idx1+1]]
        )
        
        # Coût avant dans route2
        cost_r2_before = (
            self.cost_matrix[route2.customers[idx2-1], c2] +
            self.cost_matrix[c2, route2.customers[idx2+1]]
        )
        
        # Coût après dans route2
        cost_r2_after = (
            self.cost_matrix[route2.customers[idx2-1], c1] +
            self.cost_matrix[c1, route2.customers[idx2+1]]
        )
        
        gain = (cost_r1_before + cost_r2_before) - (cost_r1_after + cost_r2_after)
        return gain