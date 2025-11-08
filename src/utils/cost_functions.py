"""
Fonctions de calcul des coûts pour le VRP

Gère le calcul composite des coûts incluant:
- Distance physique
- Coût carburant (variable selon trafic)
- Frais informels
- Péages
- Pénalités diverses
"""

import numpy as np
from typing import Dict, Tuple
from typing import List

class CostCalculator:
    """
    Calculateur de coûts multi-critères
    
    Le coût total d'une route combine plusieurs facteurs:
    - Coût kilométrique (carburant)
    - Frais fixes (péages)
    - Frais informels
    - Pénalités (stationnement, retard, surcharge, etc.)
    """
    
    def __init__(self, base_cost_matrix: np.ndarray,
                 route_data: Dict,
                 config: Dict):
        """
        Initialisation du calculateur
        
        Args:
            base_cost_matrix: Matrice de base (distances)
            route_data: Données détaillées des routes
            config: Configuration des coûts
        """
        self.base_cost_matrix = base_cost_matrix
        self.route_data = route_data
        self.config = config
        
        # Paramètres de coût
        self.fuel_cost_per_km = config.get('fuel_cost_per_km', 300)  # FCFA/km
        self.fixed_vehicle_cost = config.get('fixed_vehicle_cost', 5000)  # FCFA
        
        # Facteurs multiplicateurs selon le trafic
        self.traffic_multipliers = {
            'Faible': 1.0,
            'Moyen': 1.2,
            'Dense': 1.4,
            'Très dense': 1.6,
            'Tres dense': 1.6
        }
    
    def calculate_travel_cost(self, from_node: int, to_node: int,
                             time_period: str = 'matin') -> float:
        """
        Calcule le coût de déplacement entre deux nœuds
        
        Prend en compte:
        - Distance de base
        - Trafic selon la période
        - Frais informels
        - Péages
        
        Args:
            from_node: Nœud de départ
            to_node: Nœud d'arrivée
            time_period: 'matin', 'midi', ou 'soir'
            
        Returns:
            Coût total du trajet en FCFA
        """
        # Distance de base
        distance = self.base_cost_matrix[from_node, to_node]
        
        # Coût carburant de base
        base_fuel_cost = distance * self.fuel_cost_per_km
        
        # Ajustement selon le trafic
        traffic_key = f'Trafic {time_period.capitalize()}'
        traffic_level = self._get_traffic_level(from_node, to_node, traffic_key)
        traffic_multiplier = self.traffic_multipliers.get(traffic_level, 1.0)
        
        adjusted_fuel_cost = base_fuel_cost * traffic_multiplier
        
        # Frais informels
        informal_fees = self._get_informal_fees(from_node, to_node)
        
        # Péages
        tolls = self._get_tolls(from_node, to_node)
        
        # Coût total
        total_cost = adjusted_fuel_cost + informal_fees + tolls
        
        return total_cost
    
    def calculate_route_cost(self, route: List[int],
                            vehicle_type: str,
                            time_period: str = 'matin',
                            include_penalties: bool = True) -> Dict[str, float]:
        """
        Calcule le coût détaillé d'une route complète
        
        Args:
            route: Liste des clients [0, c1, c2, ..., cn, 0]
            vehicle_type: Type de véhicule utilisé
            time_period: Période de la journée
            include_penalties: Inclure les pénalités dans le calcul
            
        Returns:
            Dictionnaire détaillé des coûts:
            {
                'travel_cost': coût de déplacement,
                'fixed_cost': coût fixe véhicule,
                'informal_fees': frais informels totaux,
                'tolls': péages totaux,
                'penalties': pénalités diverses,
                'total_cost': coût total
            }
        """
        costs = {
            'travel_cost': 0.0,
            'fixed_cost': self.fixed_vehicle_cost,
            'informal_fees': 0.0,
            'tolls': 0.0,
            'penalties': 0.0,
            'total_distance': 0.0
        }
        
        # Calculer le coût de chaque segment
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            
            # Distance
            distance = self.base_cost_matrix[from_node, to_node]
            costs['total_distance'] += distance
            
            # Coût de déplacement
            travel_cost = self.calculate_travel_cost(from_node, to_node, time_period)
            costs['travel_cost'] += travel_cost
            
            # Frais informels
            informal = self._get_informal_fees(from_node, to_node)
            costs['informal_fees'] += informal
            
            # Péages
            tolls = self._get_tolls(from_node, to_node)
            costs['tolls'] += tolls
        
        # Ajouter les pénalités si demandé
        if include_penalties:
            # Ces pénalités seraient calculées par les gestionnaires de contraintes
            # Ici on met un placeholder
            costs['penalties'] = 0.0
        
        # Coût total
        costs['total_cost'] = (costs['travel_cost'] + 
                              costs['fixed_cost'] + 
                              costs['informal_fees'] + 
                              costs['tolls'] + 
                              costs['penalties'])
        
        return costs
    
    def calculate_insertion_cost(self, route: List[int],
                                customer: int,
                                position: int,
                                vehicle_type: str,
                                time_period: str = 'matin') -> float:
        """
        Calcule le coût d'insertion d'un client dans une route
        
        Coût d'insertion = (nouveau_coût) - (ancien_coût)
        
        Args:
            route: Route existante
            customer: Client à insérer
            position: Position d'insertion
            vehicle_type: Type de véhicule
            time_period: Période de la journée
            
        Returns:
            Coût additionnel de l'insertion
        """
        # Coût avant insertion
        cost_before = self.calculate_route_cost(
            route, vehicle_type, time_period, include_penalties=False
        )['travel_cost']
        
        # Créer route avec insertion
        new_route = route[:position] + [customer] + route[position:]
        
        # Coût après insertion
        cost_after = self.calculate_route_cost(
            new_route, vehicle_type, time_period, include_penalties=False
        )['travel_cost']
        
        # Retourner le delta
        return cost_after - cost_before
    
    def _get_traffic_level(self, from_node: int, to_node: int,
                          traffic_key: str) -> str:
        """
        Récupère le niveau de trafic pour un arc
        
        Args:
            from_node: Nœud de départ
            to_node: Nœud d'arrivée
            traffic_key: Clé du trafic ('Trafic Matin', 'Trafic Soir')
            
        Returns:
            Niveau de trafic ('Faible', 'Moyen', 'Dense', 'Très dense')
        """
        # Dans la pratique, on récupérerait depuis route_data
        # Ici, simulation simple
        if to_node == 0:  # Retour au dépôt
            return 'Faible'
        
        # Récupérer depuis les données si disponible
        key = (from_node, to_node)
        if key in self.route_data and traffic_key in self.route_data[key]:
            return self.route_data[key][traffic_key]
        
        return 'Moyen'  # Valeur par défaut
    
    def _get_informal_fees(self, from_node: int, to_node: int) -> float:
        """
        Récupère les frais informels pour un arc
        
        Args:
            from_node: Nœud de départ
            to_node: Nœud d'arrivée
            
        Returns:
            Montant des frais informels en FCFA
        """
        key = (from_node, to_node)
        if key in self.route_data and 'Frais Informels (FCFA)' in self.route_data[key]:
            return self.route_data[key]['Frais Informels (FCFA)']
        
        return 0.0  # Pas de frais par défaut
    
    def _get_tolls(self, from_node: int, to_node: int) -> float:
        """
        Récupère les péages pour un arc
        
        Args:
            from_node: Nœud de départ
            to_node: Nœud d'arrivée
            
        Returns:
            Montant des péages en FCFA
        """
        key = (from_node, to_node)
        if key in self.route_data and 'Péages (FCFA)' in self.route_data[key]:
            return self.route_data[key]['Péages (FCFA)']
        
        return 0.0  # Pas de péage par défaut
    
    def compare_time_periods(self, route: List[int],
                            vehicle_type: str) -> Dict[str, float]:
        """
        Compare les coûts d'une route selon différentes périodes
        
        Utile pour décider du meilleur moment pour une livraison
        
        Args:
            route: Route à évaluer
            vehicle_type: Type de véhicule
            
        Returns:
            Dictionnaire {periode: cout_total}
        """
        periods = ['matin', 'midi', 'soir']
        costs_by_period = {}
        
        for period in periods:
            cost_details = self.calculate_route_cost(
                route, vehicle_type, period, include_penalties=False
            )
            costs_by_period[period] = cost_details['total_cost']
        
        return costs_by_period
    
    def get_cheapest_time_period(self, route: List[int],
                                vehicle_type: str) -> Tuple[str, float]:
        """
        Détermine la période la moins coûteuse pour une route
        
        Args:
            route: Route à évaluer
            vehicle_type: Type de véhicule
            
        Returns:
            (meilleure_periode, cout_minimal)
        """
        costs_by_period = self.compare_time_periods(route, vehicle_type)
        best_period = min(costs_by_period, key=costs_by_period.get)
        best_cost = costs_by_period[best_period]
        
        return best_period, best_cost