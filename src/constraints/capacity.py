"""
Gestion des contraintes de capacité des véhicules

Chaque véhicule a une capacité maximale (en poids, volume, ou unités).
La charge totale de chaque route ne doit pas dépasser cette capacité.
"""

from typing import Dict, List, Tuple
from src.models.solution import Route, Solution

class CapacityManager:
    """
    Gestionnaire des contraintes de capacité
    
    Responsabilités:
    - Vérifier le respect des capacités
    - Calculer les charges des routes
    - Détecter les violations
    - Proposer des corrections
    """
    
    def __init__(self, demands: Dict[int, float],
                 vehicle_capacities: Dict[str, float],
                 config: Dict):
        """
        Initialisation du gestionnaire de capacité
        
        Args:
            demands: Dictionnaire {client_id: demande}
            vehicle_capacities: Dictionnaire {type_vehicule: capacite}
            config: Configuration générale
        """
        self.demands = demands
        self.vehicle_capacities = vehicle_capacities
        self.config = config
        
        # Pénalité pour dépassement de capacité
        self.overload_penalty = config.get('overload_penalty', 50000)
    
    def calculate_route_load(self, route: List[int]) -> float:
        """
        Calcule la charge totale d'une route
        
        Args:
            route: Liste des clients [0, c1, c2, ..., cn, 0]
            
        Returns:
            Charge totale (somme des demandes)
        """
        total_load = 0.0
        
        for customer in route:
            if customer == 0:  # Ignorer le dépôt
                continue
            total_load += self.demands.get(customer, 0.0)
        
        return total_load
    
    def is_route_feasible(self, route: Route) -> Tuple[bool, float]:
        """
        Vérifie si une route respecte la contrainte de capacité
        
        Args:
            route: Objet Route à vérifier
            
        Returns:
            (faisabilite, surcharge)
            - faisabilite: True si charge <= capacité
            - surcharge: quantité en excès (0 si faisable)
        """
        load = self.calculate_route_load(route.customers)
        overload = max(0, load - route.capacity)
        
        is_feasible = (overload == 0)
        
        return is_feasible, overload
    
    def is_solution_feasible(self, solution: Solution) -> Tuple[bool, List[str]]:
        """
        Vérifie si toutes les routes d'une solution respectent les capacités
        
        Args:
            solution: Solution complète à vérifier
            
        Returns:
            (faisabilite, liste_violations)
        """
        violations = []
        all_feasible = True
        
        for route in solution.routes:
            is_feasible, overload = self.is_route_feasible(route)
            
            if not is_feasible:
                all_feasible = False
                violations.append(
                    f"Route {route.vehicle_id}: Surcharge de {overload:.1f} unites "
                    f"(Charge: {route.load:.1f}, Capacite: {route.capacity:.1f})"
                )
        
        return all_feasible, violations
    
    def calculate_capacity_penalty(self, solution: Solution) -> float:
        """
        Calcule la pénalité totale pour dépassement de capacité
        
        Args:
            solution: Solution à évaluer
            
        Returns:
            Pénalité totale en FCFA
        """
        total_penalty = 0.0
        
        for route in solution.routes:
            _, overload = self.is_route_feasible(route)
            
            if overload > 0:
                # Pénalité proportionnelle au dépassement
                total_penalty += overload * self.overload_penalty
        
        return total_penalty
    
    def can_insert_customer(self, route: Route, customer: int) -> bool:
        """
        Vérifie si un client peut être inséré dans une route
        sans violer la capacité
        
        Args:
            route: Route existante
            customer: ID du client à insérer
            
        Returns:
            True si l'insertion est possible
        """
        current_load = self.calculate_route_load(route.customers)
        customer_demand = self.demands.get(customer, 0.0)
        new_load = current_load + customer_demand
        
        return new_load <= route.capacity
    
    def get_remaining_capacity(self, route: Route) -> float:
        """
        Calcule la capacité restante d'une route
        
        Args:
            route: Route à analyser
            
        Returns:
            Capacité restante (capacité - charge actuelle)
        """
        current_load = self.calculate_route_load(route.customers)
        return route.capacity - current_load
    
    def split_overloaded_route(self, route: Route) -> List[Route]:
        """
        Divise une route en surcharge en plusieurs routes faisables
        
        Algorithme:
        1. Créer une nouvelle route vide
        2. Ajouter les clients un par un
        3. Si ajout dépasse capacité, créer une nouvelle route
        
        Args:
            route: Route en surcharge
            
        Returns:
            Liste de routes faisables
        """
        customers = [c for c in route.customers if c != 0]
        new_routes = []
        
        current_route = Route(
            vehicle_id=route.vehicle_id,
            vehicle_type=route.vehicle_type,
            capacity=route.capacity
        )
        current_route.customers = [0]  # Commencer au dépôt
        current_load = 0.0
        
        for customer in customers:
            demand = self.demands.get(customer, 0.0)
            
            if current_load + demand <= route.capacity:
                # Ajouter à la route courante
                current_route.customers.append(customer)
                current_load += demand
            else:
                # Finaliser la route courante
                current_route.customers.append(0)  # Retour au dépôt
                current_route.load = current_load
                new_routes.append(current_route)
                
                # Créer une nouvelle route
                current_route = Route(
                    vehicle_id=len(new_routes),
                    vehicle_type=route.vehicle_type,
                    capacity=route.capacity
                )
                current_route.customers = [0, customer]
                current_load = demand
        
        # Finaliser la dernière route
        if len(current_route.customers) > 1:
            current_route.customers.append(0)
            current_route.load = current_load
            new_routes.append(current_route)
        
        return new_routes
    
    def balance_loads(self, solution: Solution) -> Solution:
        """
        Tente d'équilibrer les charges entre les routes
        
        Objectif: éviter d'avoir des routes très chargées et d'autres presque vides
        
        Stratégie:
        1. Identifier les routes les plus chargées et les moins chargées
        2. Transférer des clients des routes chargées vers les routes légères
        
        Args:
            solution: Solution à équilibrer
            
        Returns:
            Solution avec charges équilibrées
        """
        new_solution = solution.copy()
        
        # Trier les routes par charge (décroissant)
        routes_by_load = sorted(
            new_solution.routes,
            key=lambda r: self.calculate_route_load(r.customers),
            reverse=True
        )
        
        # Essayer de transférer des clients des routes chargées vers les légères
        for heavy_route in routes_by_load[:len(routes_by_load)//2]:
            for light_route in routes_by_load[len(routes_by_load)//2:]:
                
                # Chercher un client à transférer
                for customer in heavy_route.get_customers_only():
                    demand = self.demands.get(customer, 0.0)
                    
                    # Vérifier si le transfert est possible
                    if self.can_insert_customer(light_route, customer):
                        heavy_load = self.calculate_route_load(heavy_route.customers)
                        light_load = self.calculate_route_load(light_route.customers)
                        
                        # Transférer si cela améliore l'équilibre
                        if abs((heavy_load - demand) - (light_load + demand)) < abs(heavy_load - light_load):
                            heavy_route.remove_customer(customer)
                            light_route.add_customer(customer, len(light_route.customers) - 1)
                            break
        
        return new_solution
    
    def suggest_vehicle_type(self, total_demand: float) -> str:
        """
        Suggère le type de véhicule approprié pour une demande donnée
        
        Args:
            total_demand: Demande totale à transporter
            
        Returns:
            Type de véhicule recommandé
        """
        # Trier les véhicules par capacité croissante
        sorted_vehicles = sorted(
            self.vehicle_capacities.items(),
            key=lambda x: x[1]
        )
        
        # Choisir le plus petit véhicule qui peut contenir la demande
        for vehicle_type, capacity in sorted_vehicles:
            if capacity >= total_demand:
                return vehicle_type
        
        # Si aucun véhicule n'est assez grand, retourner le plus grand
        return sorted_vehicles[-1][0]