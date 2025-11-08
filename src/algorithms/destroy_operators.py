"""
Opérateurs de destruction pour ALNS
Retirent des clients des routes pour permettre la réoptimisation
"""

import numpy as np
import random
from typing import List, Set
from src.models.solution import Solution

class DestroyOperators:
    """Collection d'opérateurs de destruction"""
    
    def __init__(self, config: dict):
        self.config = config
        self.random_state = np.random.RandomState(config.get('seed', 42))
        
    def random_removal(self, solution: Solution, q: int) -> tuple[Solution, List[int]]:
        """
        Retire q clients aléatoirement
        
        Args:
            solution: Solution courante
            q: Nombre de clients à retirer
            
        Returns:
            (nouvelle_solution, clients_retirés)
        """
        new_solution = solution.copy()
        all_customers = new_solution.get_all_customers()
        
        if len(all_customers) == 0:
            return new_solution, []
        
        q = min(q, len(all_customers))
        removed_customers = self.random_state.choice(all_customers, size=q, replace=False).tolist()
        
        # Retirer les clients des routes
        for route in new_solution.routes:
            for customer in removed_customers:
                route.remove_customer(customer)
        
        new_solution.remove_empty_routes()
        new_solution.unassigned_customers.extend(removed_customers)
        
        return new_solution, removed_customers
    
    def worst_removal(self, solution: Solution, q: int, 
                      cost_matrix: np.ndarray) -> tuple[Solution, List[int]]:
        """
        Retire les q clients les plus coûteux
        Coût = économie réalisée en retirant le client
        """
        new_solution = solution.copy()
        all_customers = new_solution.get_all_customers()
        
        if len(all_customers) == 0:
            return new_solution, []
        
        # Calculer le coût de retrait pour chaque client
        removal_costs = {}
        
        for route in new_solution.routes:
            customers = route.get_customers_only()
            for i, customer in enumerate(customers):
                # Coût avant retrait
                if i == 0:
                    prev_node = 0
                else:
                    prev_node = customers[i-1]
                
                if i == len(customers) - 1:
                    next_node = 0
                else:
                    next_node = customers[i+1]
                
                cost_before = (cost_matrix[prev_node, customer] + 
                              cost_matrix[customer, next_node])
                cost_after = cost_matrix[prev_node, next_node]
                
                # Économie = coût évité
                saving = cost_before - cost_after
                removal_costs[customer] = -saving  # Négatif car on veut retirer les pires
        
        # Trier par coût de retrait (plus bas = pire)
        sorted_customers = sorted(removal_costs.keys(), 
                                 key=lambda x: removal_costs[x])
        
        q = min(q, len(sorted_customers))
        removed_customers = sorted_customers[:q]
        
        # Retirer les clients
        for route in new_solution.routes:
            for customer in removed_customers:
                route.remove_customer(customer)
        
        new_solution.remove_empty_routes()
        new_solution.unassigned_customers.extend(removed_customers)
        
        return new_solution, removed_customers
    
    def shaw_removal(self, solution: Solution, q: int,
                    cost_matrix: np.ndarray,
                    time_windows: dict) -> tuple[Solution, List[int]]:
        """
        Retire q clients similaires (proximité géographique + temporelle)
        Shaw removal - très efficace pour VRP
        """
        new_solution = solution.copy()
        all_customers = new_solution.get_all_customers()
        
        if len(all_customers) == 0:
            return new_solution, []
        
        # Choisir seed client
        seed = self.random_state.choice(all_customers)
        removed_customers = [seed]
        
        # Paramètres de similarité
        alpha = 0.5  # Poids distance
        beta = 0.3   # Poids temps
        gamma = 0.2  # Poids contraintes
        
        while len(removed_customers) < q and len(removed_customers) < len(all_customers):
            # Calculer similarité avec tous les clients non retirés
            similarities = {}
            
            for customer in all_customers:
                if customer in removed_customers:
                    continue
                
                # Distance géographique moyenne
                dist_sim = np.mean([cost_matrix[customer, r] 
                                   for r in removed_customers])
                
                # Similarité temporelle
                tw_customer = time_windows.get(customer, (0, 24))
                tw_seed = time_windows.get(seed, (0, 24))
                time_sim = abs(tw_customer[0] - tw_seed[0])
                
                # Normaliser
                dist_sim = dist_sim / np.max(cost_matrix)
                time_sim = time_sim / 24.0
                
                similarity = alpha * dist_sim + beta * time_sim
                similarities[customer] = similarity
            
            if not similarities:
                break
            
            # Choisir client le plus similaire (probabiliste)
            sorted_customers = sorted(similarities.keys(), 
                                     key=lambda x: similarities[x])
            
            # Sélection probabiliste (favorise les plus similaires)
            probabilities = np.exp(-np.array([similarities[c] for c in sorted_customers]))
            probabilities /= probabilities.sum()
            
            next_customer = self.random_state.choice(sorted_customers, p=probabilities)
            removed_customers.append(next_customer)
        
        # Retirer les clients
        for route in new_solution.routes:
            for customer in removed_customers:
                route.remove_customer(customer)
        
        new_solution.remove_empty_routes()
        new_solution.unassigned_customers.extend(removed_customers)
        
        return new_solution, removed_customers
    
    def route_removal(self, solution: Solution, q: int) -> tuple[Solution, List[int]]:
        """
        Retire une route complète (tous ses clients)
        Utile pour restructuration majeure
        """
        new_solution = solution.copy()
        
        if len(new_solution.routes) == 0:
            return new_solution, []
        
        # Choisir une route aléatoirement
        route_to_remove = self.random_state.choice(new_solution.routes)
        removed_customers = route_to_remove.get_customers_only()
        
        # Limiter à q clients
        if len(removed_customers) > q:
            removed_customers = self.random_state.choice(removed_customers, 
                                                        size=q, replace=False).tolist()
        
        # Retirer les clients
        for customer in removed_customers:
            route_to_remove.remove_customer(customer)
        
        new_solution.remove_empty_routes()
        new_solution.unassigned_customers.extend(removed_customers)
        
        return new_solution, removed_customers
    
    def cluster_removal(self, solution: Solution, q: int,
                       cost_matrix: np.ndarray) -> tuple[Solution, List[int]]:
        """
        Retire un cluster géographique de clients
        """
        new_solution = solution.copy()
        all_customers = new_solution.get_all_customers()
        
        if len(all_customers) == 0:
            return new_solution, []
        
        # Choisir centre du cluster
        center = self.random_state.choice(all_customers)
        
        # Trouver q clients les plus proches
        distances = [(c, cost_matrix[center, c]) for c in all_customers if c != center]
        distances.sort(key=lambda x: x[1])
        
        q = min(q - 1, len(distances))  # -1 pour inclure le centre
        removed_customers = [center] + [c for c, d in distances[:q]]
        
        # Retirer les clients
        for route in new_solution.routes:
            for customer in removed_customers:
                route.remove_customer(customer)
        
        new_solution.remove_empty_routes()
        new_solution.unassigned_customers.extend(removed_customers)
        
        return new_solution, removed_customers