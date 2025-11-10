"""
Opérateurs de réparation pour ALNS
Réinsèrent les clients retirés dans les routes
"""

import numpy as np
from typing import List, Dict, Tuple
from src.models.solution import Solution, Route

class RepairOperators:
    """Collection d'opérateurs de réparation"""
    
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
        self.random_state = np.random.RandomState(config.get('seed', 42))
    
    def greedy_insertion(self, solution: Solution) -> Solution:
        """
        Insertion gloutonne: insère chaque client à la meilleure position
        """
        new_solution = solution.copy()
        unassigned = new_solution.unassigned_customers.copy()
        
        for customer in unassigned:
            best_position = None
            best_route = None
            best_cost = float('inf')
            
            # Essayer toutes les routes existantes
            for route in new_solution.routes:
                for pos in range(1, len(route.customers)):
                    cost, feasible = self._calculate_insertion_cost(
                        customer, route, pos
                    )
                    
                    if feasible and cost < best_cost:
                        best_cost = cost
                        best_position = pos
                        best_route = route
            
            # Si aucune position trouvée, créer nouvelle route
            if best_route is None:
                if len(new_solution.routes) < self.config.get('max_vehicles', 20):
                    new_route = self._create_new_route(customer)
                    new_solution.add_route(new_route)
                    new_solution.unassigned_customers.remove(customer)
            else:
                best_route.add_customer(customer, best_position)
                new_solution.unassigned_customers.remove(customer)
        
        return new_solution
    
    def regret_insertion(self, solution: Solution, k: int = 3) -> Solution:
        """
        Insertion avec regret-k
        Favorise les clients difficiles à insérer
        
        Args:
            k: nombre de meilleures positions à considérer
        """
        new_solution = solution.copy()
        
        while new_solution.unassigned_customers:
            max_regret = -float('inf')
            best_customer = None
            best_insertion = None
            
            # Pour chaque client non assigné
            for customer in new_solution.unassigned_customers:
                # Trouver k meilleures insertions
                insertions = []
                
                for route in new_solution.routes:
                    for pos in range(1, len(route.customers)):
                        cost, feasible = self._calculate_insertion_cost(
                            customer, route, pos
                        )
                        
                        if feasible:
                            insertions.append((cost, route, pos))
                
                # Possibilité de nouvelle route
                if len(new_solution.routes) < self.config.get('max_vehicles', 20):
                    new_route_cost = self._estimate_new_route_cost(customer)
                    insertions.append((new_route_cost, None, -1))
                
                if not insertions:
                    continue
                
                # Trier par coût
                insertions.sort(key=lambda x: x[0])
                
                # Calculer regret
                if len(insertions) >= k:
                    regret = sum(insertions[i][0] - insertions[0][0] 
                               for i in range(1, k))
                else:
                    regret = sum(insertions[i][0] - insertions[0][0] 
                               for i in range(1, len(insertions)))
                
                # Mettre à jour meilleur regret
                if regret > max_regret:
                    max_regret = regret
                    best_customer = customer
                    best_insertion = insertions[0]
            
            # Insérer le client avec le plus grand regret
            if best_customer is None:
                break
            
            cost, route, pos = best_insertion
            
            if route is None:
                # Créer nouvelle route
                new_route = self._create_new_route(best_customer)
                new_solution.add_route(new_route)
            else:
                route.add_customer(best_customer, pos)
            
            new_solution.unassigned_customers.remove(best_customer)
        
        return new_solution
    
    def time_oriented_insertion(self, solution: Solution) -> Solution:
        """
        Insertion orientée temps: respecte l'ordre chronologique
        """
        new_solution = solution.copy()
        unassigned = new_solution.unassigned_customers.copy()
        
        # Trier clients par début de fenêtre temporelle
        sorted_customers = sorted(
            unassigned,
            key=lambda c: self.time_windows.get(c, (0, 24))[0]
        )
        
        for customer in sorted_customers:
            inserted = False
            tw_start, tw_end = self.time_windows.get(customer, (0, 24))
            
            # Essayer d'insérer dans routes existantes
            for route in new_solution.routes:
                # Trouver position temporellement compatible
                for pos in range(1, len(route.customers)):
                    prev_customer = route.customers[pos - 1]
                    next_customer = route.customers[pos]
                    
                    # Vérifier compatibilité temporelle
                    arrival_time = self._estimate_arrival_time(
                        prev_customer, customer, route
                    )
                    
                    if arrival_time <= tw_end + self.config.get('time_tolerance', 1.0):
                        _, feasible = self._calculate_insertion_cost(
                            customer, route, pos
                        )
                        
                        if feasible:
                            route.add_customer(customer, pos)
                            new_solution.unassigned_customers.remove(customer)
                            inserted = True
                            break
                
                if inserted:
                    break
            
            # Créer nouvelle route si nécessaire
            if not inserted:
                if len(new_solution.routes) < self.config.get('max_vehicles', 20):
                    new_route = self._create_new_route(customer)
                    new_solution.add_route(new_route)
                    new_solution.unassigned_customers.remove(customer)
        
        return new_solution
    
    def random_insertion(self, solution: Solution) -> Solution:
        """
        Insertion aléatoire avec vérification de faisabilité
        Utile pour diversification
        """
        new_solution = solution.copy()
        unassigned = new_solution.unassigned_customers.copy()
        self.random_state.shuffle(unassigned)
        
        for customer in unassigned:
            feasible_insertions = []
            
            # Trouver toutes les insertions faisables
            for route in new_solution.routes:
                for pos in range(1, len(route.customers)):
                    _, feasible = self._calculate_insertion_cost(
                        customer, route, pos
                    )
                    
                    if feasible:
                        feasible_insertions.append((route, pos))
            
            # Choisir aléatoirement parmi les positions faisables
            if feasible_insertions:
                idx = self.random_state.randint(0, len(feasible_insertions))
                route, pos = feasible_insertions[idx]
                route.add_customer(customer, pos)
                new_solution.unassigned_customers.remove(customer)
            else:
                # Créer nouvelle route
                if len(new_solution.routes) < self.config.get('max_vehicles', 20):
                    new_route = self._create_new_route(customer)
                    new_solution.add_route(new_route)
                    new_solution.unassigned_customers.remove(customer)
        
        return new_solution
    
    def _calculate_insertion_cost(self, customer: int, route: Route, 
                                  position: int) -> Tuple[float, bool]:
        """
        Calcule le coût d'insertion et vérifie la faisabilité
        
        Returns:
            (coût_insertion, faisabilité)
        """
        prev_customer = route.customers[position - 1]
        next_customer = route.customers[position]
        
        # Coût avant insertion
        cost_before = self.cost_matrix[prev_customer, next_customer]
        
        # Coût après insertion
        cost_after = (self.cost_matrix[prev_customer, customer] + 
                     self.cost_matrix[customer, next_customer])
        
        insertion_cost = cost_after - cost_before
        
        # Vérifier capacité
        new_load = route.load + self.demands.get(customer, 0)
        if new_load > route.capacity:
            return insertion_cost, False
        
        # Vérifier time windows (simplifié)
        tw_start, tw_end = self.time_windows.get(customer, (0, 24))
        arrival_time = self._estimate_arrival_time(prev_customer, customer, route)
        
        if arrival_time > tw_end + self.config.get('time_tolerance', 1.0):
            return insertion_cost, False
        
        return insertion_cost, True
    
    def _estimate_arrival_time(self, from_customer: int, 
                              to_customer: int, route: Route) -> float:
        """Estime le temps d'arrivée"""
        distance = self.cost_matrix[from_customer, to_customer]
        avg_speed = self.config.get('average_speed', 25)
        travel_time = distance / avg_speed
        
        # Temps depuis le début de la route (simplifié)
        start_time = self.config.get('start_time', 6.0)
        return start_time + travel_time
    
    def _create_new_route(self, customer: int) -> Route:
        """Crée une nouvelle route avec un seul client"""
        vehicle_type = "Camionnette"  # Par défaut
        capacity = self.vehicle_capacities[vehicle_type]
        
        route = Route(
            vehicle_id=len([0]),  # ID temporaire
            vehicle_type=vehicle_type,
            capacity=capacity
        )
        route.customers = [0, customer, 0]
        route.load = self.demands.get(customer, 0)
        
        return route
    
    def _estimate_new_route_cost(self, customer: int) -> float:
        """Estime le coût de création d'une nouvelle route"""
        fixed_cost = self.config.get('fixed_vehicle_cost', 5000)
        travel_cost = 2 * self.cost_matrix[0, customer]  # Aller-retour
        return fixed_cost + travel_cost