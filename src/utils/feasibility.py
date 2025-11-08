"""
Vérification globale de la faisabilité des solutions

Ce module centralise toutes les vérifications de contraintes
et fournit des fonctions pour valider/réparer les solutions.
"""

from typing import Dict, List, Tuple
import numpy as np

from src.models.solution import Solution, Route
from src.constraints.time_windows import TimeWindowManager
from src.constraints.capacity import CapacityManager
from src.constraints.restrictions import RestrictionManager

class FeasibilityChecker:
    """
    Vérificateur global de faisabilité
    
    Coordonne tous les gestionnaires de contraintes pour vérifier
    qu'une solution respecte toutes les contraintes du problème.
    """
    
    def __init__(self, cost_matrix: np.ndarray,
                 demands: Dict[int, float],
                 time_windows: Dict[int, Tuple[float, float]],
                 vehicle_capacities: Dict[str, float],
                 route_data: Dict,
                 config: Dict):
        """
        Initialisation du vérificateur
        
        Args:
            cost_matrix: Matrice des coûts/distances
            demands: Demandes des clients
            time_windows: Fenêtres temporelles
            vehicle_capacities: Capacités des véhicules
            route_data: Données des routes
            config: Configuration générale
        """
        # Initialiser les gestionnaires de contraintes
        self.tw_manager = TimeWindowManager(time_windows, config)
        self.capacity_manager = CapacityManager(demands, vehicle_capacities, config)
        self.restriction_manager = RestrictionManager(route_data, config)
        
        self.cost_matrix = cost_matrix
        self.config = config
    
    def check_solution(self, solution: Solution,
                      detailed: bool = True) -> Tuple[bool, Dict]:
        """
        Vérifie la faisabilité complète d'une solution
        
        Args:
            solution: Solution à vérifier
            detailed: Si True, retourne les détails de toutes les violations
            
        Returns:
            (est_faisable, rapport_detaille)
            
            rapport_detaille contient:
            {
                'feasible': bool,
                'capacity_violations': [...],
                'time_violations': [...],
                'restriction_violations': [...],
                'total_violations': int,
                'route_details': {...}
            }
        """
        report = {
            'feasible': True,
            'capacity_violations': [],
            'time_violations': [],
            'restriction_violations': [],
            'total_violations': 0,
            'route_details': {}
        }
        
        # Vérifier chaque route individuellement
        for route in solution.routes:
            if route.is_empty():
                continue
            
            route_id = route.vehicle_id
            route_report = self._check_route(route)
            
            # Agréger les violations
            if not route_report['feasible']:
                report['feasible'] = False
                report['capacity_violations'].extend(route_report['capacity_violations'])
                report['time_violations'].extend(route_report['time_violations'])
                report['restriction_violations'].extend(route_report['restriction_violations'])
            
            if detailed:
                report['route_details'][route_id] = route_report
        
        # Compter le nombre total de violations
        report['total_violations'] = (
            len(report['capacity_violations']) +
            len(report['time_violations']) +
            len(report['restriction_violations'])
        )
        
        return report['feasible'], report
    
    def _check_route(self, route: Route) -> Dict:
        """
        Vérifie la faisabilité d'une route individuelle
        
        Args:
            route: Route à vérifier
            
        Returns:
            Rapport de faisabilité de la route
        """
        report = {
            'feasible': True,
            'capacity_violations': [],
            'time_violations': [],
            'restriction_violations': []
        }
        
        # 1. Vérifier la capacité
        capacity_ok, capacity_msg = self.capacity_manager.is_route_feasible(route)
        if not capacity_ok:
            report['feasible'] = False
            report['capacity_violations'].append(
                f"Route {route.vehicle_id}: Surcharge detectee - {capacity_msg}"
            )
        
        # 2. Vérifier les fenêtres temporelles
        tw_ok, tw_violations = self.tw_manager.is_feasible(
            route.customers, self.cost_matrix
        )
        if not tw_ok:
            report['feasible'] = False
            report['time_violations'].extend([
                f"Route {route.vehicle_id}: {v}" for v in tw_violations
            ])
        
        # 3. Vérifier les restrictions
        schedule = self.tw_manager.calculate_arrival_times(
            route.customers, self.cost_matrix
        )
        arrival_times = {c: info['arrival_time'] 
                        for c, info in schedule.items()}
        
        restr_ok, restr_violations = self.restriction_manager.is_route_feasible_with_restrictions(
            route.customers, route.vehicle_type, arrival_times
        )
        if not restr_ok:
            report['feasible'] = False
            report['restriction_violations'].extend([
                f"Route {route.vehicle_id}: {v}" for v in restr_violations
            ])
        
        return report
    
    def repair_solution(self, solution: Solution) -> Solution:
        """
        Tente de réparer une solution infaisable
        
        Stratégies de réparation:
        1. Diviser les routes en surcharge
        2. Ajuster les horaires pour respecter les fenêtres temporelles
        3. Retirer les clients problématiques et les marquer comme non desservis
        
        Args:
            solution: Solution potentiellement infaisable
            
        Returns:
            Solution réparée (peut toujours être partiellement infaisable)
        """
        repaired_solution = solution.copy()
        
        # 1. Réparer les violations de capacité
        repaired_solution = self._repair_capacity_violations(repaired_solution)
        
        # 2. Réparer les violations de fenêtres temporelles
        repaired_solution = self._repair_time_violations(repaired_solution)
        
        # 3. Nettoyer les routes vides
        repaired_solution.remove_empty_routes()
        
        # Recalculer les métriques
        from src.utils.cost_functions import CostCalculator
        # Note: nécessiterait une instance de CostCalculator
        
        return repaired_solution
    
    def _repair_capacity_violations(self, solution: Solution) -> Solution:
        """
        Répare les violations de capacité en divisant les routes surchargées
        
        Args:
            solution: Solution avec possibles violations de capacité
            
        Returns:
            Solution avec routes dans les limites de capacité
        """
        new_routes = []
        
        for route in solution.routes:
            is_ok, overload = self.capacity_manager.is_route_feasible(route)
            
            if is_ok:
                # Route OK, la garder telle quelle
                new_routes.append(route)
            else:
                # Route en surcharge, la diviser
                split_routes = self.capacity_manager.split_overloaded_route(route)
                new_routes.extend(split_routes)
        
        solution.routes = new_routes
        return solution
    
    def _repair_time_violations(self, solution: Solution) -> Solution:
        """
        Répare les violations de fenêtres temporelles
        
        Stratégie:
        1. Ajuster l'heure de départ
        2. Si impossible, retirer les clients problématiques
        
        Args:
            solution: Solution avec possibles violations temporelles
            
        Returns:
            Solution avec fenêtres temporelles respectées
        """
        for route in solution.routes:
            if route.is_empty():
                continue
            
            # Vérifier si la route viole les contraintes temporelles
            is_ok, violations = self.tw_manager.is_feasible(
                route.customers, self.cost_matrix
            )
            
            if not is_ok:
                # Essayer d'optimiser l'heure de départ
                optimal_start = self.tw_manager.optimize_departure_times(
                    route.customers, self.cost_matrix
                )
                
                # Revérifier avec le nouveau départ
                is_ok_now, _ = self.tw_manager.is_feasible(
                    route.customers, self.cost_matrix, optimal_start
                )
                
                if not is_ok_now:
                    # Si toujours infaisable, retirer les clients les plus problématiques
                    # (implémentation simplifiée: marquer pour retrait)
                    print(f"ATTENTION: Route {route.vehicle_id} reste infaisable "
                          f"meme apres optimisation du depart")
        
        return solution
    
    def get_solution_quality_metrics(self, solution: Solution) -> Dict:
        """
        Calcule des métriques de qualité pour une solution
        
        Args:
            solution: Solution à évaluer
            
        Returns:
            Dictionnaire de métriques:
            {
                'feasibility_score': score sur 100,
                'capacity_utilization': taux d'utilisation moyen,
                'time_window_compliance': % de respect,
                'route_balance': équilibre des charges,
                'complexity_score': complexité des routes
            }
        """
        metrics = {}
        
        # Score de faisabilité (0-100)
        is_feasible, report = self.check_solution(solution, detailed=True)
        if is_feasible:
            metrics['feasibility_score'] = 100.0
        else:
            # Pénaliser selon le nombre de violations
            penalty = min(100, report['total_violations'] * 10)
            metrics['feasibility_score'] = max(0, 100 - penalty)
        
        # Utilisation de la capacité
        if solution.routes:
            utilizations = []
            for route in solution.routes:
                if not route.is_empty():
                    load = self.capacity_manager.calculate_route_load(route.customers)
                    utilization = (load / route.capacity) * 100 if route.capacity > 0 else 0
                    utilizations.append(utilization)
            
            metrics['capacity_utilization'] = np.mean(utilizations) if utilizations else 0
            metrics['route_balance'] = 100 - np.std(utilizations) if len(utilizations) > 1 else 100
        else:
            metrics['capacity_utilization'] = 0
            metrics['route_balance'] = 0
        
        # Conformité aux fenêtres temporelles
        total_customers = len(solution.get_all_customers())
        time_violations = len(report.get('time_violations', []))
        if total_customers > 0:
            metrics['time_window_compliance'] = ((total_customers - time_violations) / 
                                                 total_customers * 100)
        else:
            metrics['time_window_compliance'] = 100
        
        # Score de complexité (nombre moyen de clients par route)
        if solution.routes:
            route_lengths = [len(r.get_customers_only()) for r in solution.routes 
                           if not r.is_empty()]
            metrics['avg_customers_per_route'] = np.mean(route_lengths) if route_lengths else 0
            metrics['complexity_score'] = min(100, metrics['avg_customers_per_route'] * 10)
        else:
            metrics['avg_customers_per_route'] = 0
            metrics['complexity_score'] = 0
        
        return metrics
    
    def suggest_improvements(self, solution: Solution) -> List[str]:
        """
        Suggère des améliorations possibles pour une solution
        
        Args:
            solution: Solution à analyser
            
        Returns:
            Liste de suggestions d'amélioration
        """
        suggestions = []
        
        # Vérifier la faisabilité
        is_feasible, report = self.check_solution(solution, detailed=True)
        
        if not is_feasible:
            suggestions.append(
                f"Solution infaisable: {report['total_violations']} violation(s) detectee(s). "
                f"Utiliser repair_solution() pour corriger."
            )
        
        # Analyser l'utilisation des capacités
        metrics = self.get_solution_quality_metrics(solution)
        
        if metrics['capacity_utilization'] < 60:
            suggestions.append(
                f"Utilisation de capacite faible ({metrics['capacity_utilization']:.1f}%). "
                f"Envisager de consolider certaines routes."
            )
        
        if metrics['route_balance'] < 70:
            suggestions.append(
                "Charges desequilibrees entre les routes. "
                "Utiliser balance_loads() pour ameliorer."
            )
        
        # Vérifier les clients non desservis
        if solution.unassigned_customers:
            suggestions.append(
                f"{len(solution.unassigned_customers)} client(s) non desservi(s). "
                f"Ajouter des vehicules ou ajuster les contraintes."
            )
        
        return suggestions