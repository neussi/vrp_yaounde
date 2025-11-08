"""
Gestion des contraintes de fenêtres temporelles (Time Windows)

Les fenêtres temporelles définissent les périodes durant lesquelles
chaque client peut être visité. Le respect de ces contraintes est
crucial pour la faisabilité de la solution.

Format: time_window = (earliest_start, latest_end)
Exemple: (8.0, 12.0) signifie que le client doit être visité entre 8h et 12h
"""

from typing import Dict, Tuple, List
import numpy as np

class TimeWindowManager:
    """
    Gestionnaire des contraintes de fenêtres temporelles
    
    Responsabilités:
    - Vérifier le respect des fenêtres temporelles
    - Calculer les temps d'arrivée
    - Gérer les temps d'attente
    - Calculer les pénalités de retard
    """
    
    def __init__(self, time_windows: Dict[int, Tuple[float, float]],
                 config: Dict):
        """
        Initialisation du gestionnaire
        
        Args:
            time_windows: Dictionnaire {client_id: (debut, fin)}
            config: Configuration (service_time, average_speed, etc.)
        """
        self.time_windows = time_windows
        self.config = config
        
        # Paramètres par défaut
        self.service_time = config.get('service_time', 0.5)  # 30 minutes
        self.average_speed = config.get('average_speed', 25)  # 25 km/h
        self.time_tolerance = config.get('time_tolerance', 1.0)  # 1 heure de tolérance
        self.late_penalty = config.get('late_penalty', 10000)  # Pénalité par heure de retard
    
    def calculate_arrival_times(self, route: List[int], 
                                cost_matrix: np.ndarray,
                                start_time: float = 6.0) -> Dict[int, Dict]:
        """
        Calcule les temps d'arrivée pour tous les clients d'une route
        
        Prend en compte:
        - Le temps de trajet entre clients
        - Les temps d'attente (si arrivée avant début de fenêtre)
        - Les temps de service
        
        Args:
            route: Liste des clients dans l'ordre [0, c1, c2, ..., cn, 0]
            cost_matrix: Matrice des distances/coûts
            start_time: Heure de départ du dépôt (par défaut 6h du matin)
            
        Returns:
            Dictionnaire {client_id: {
                'arrival_time': temps_arrivee,
                'start_service': debut_service,
                'departure_time': temps_depart,
                'waiting_time': temps_attente,
                'lateness': retard
            }}
        """
        schedule = {}
        current_time = start_time
        
        for i in range(len(route) - 1):
            from_node = route[i]
            to_node = route[i + 1]
            
            # Calculer le temps de trajet
            distance = cost_matrix[from_node, to_node]
            travel_time = distance / self.average_speed
            
            # Temps d'arrivée au client
            arrival_time = current_time + travel_time
            
            # Récupérer la fenêtre temporelle du client
            if to_node == 0:  # Retour au dépôt
                tw_start, tw_end = 0, 24
            else:
                tw_start, tw_end = self.time_windows.get(to_node, (0, 24))
            
            # Calculer le temps d'attente si arrivée trop tôt
            if arrival_time < tw_start:
                waiting_time = tw_start - arrival_time
                start_service = tw_start
            else:
                waiting_time = 0
                start_service = arrival_time
            
            # Calculer le retard si arrivée trop tard
            if arrival_time > tw_end:
                lateness = arrival_time - tw_end
            else:
                lateness = 0
            
            # Temps de départ après service
            departure_time = start_service + self.service_time
            
            # Enregistrer les informations
            schedule[to_node] = {
                'arrival_time': arrival_time,
                'start_service': start_service,
                'departure_time': departure_time,
                'waiting_time': waiting_time,
                'lateness': lateness
            }
            
            # Mettre à jour le temps courant
            current_time = departure_time
        
        return schedule
    
    def is_feasible(self, route: List[int], 
                   cost_matrix: np.ndarray,
                   start_time: float = 6.0) -> Tuple[bool, List[str]]:
        """
        Vérifie si une route respecte toutes les contraintes temporelles
        
        Args:
            route: Liste des clients [0, c1, c2, ..., cn, 0]
            cost_matrix: Matrice des distances
            start_time: Heure de départ
            
        Returns:
            (faisabilite, liste_violations)
            - faisabilite: True si la route est faisable
            - liste_violations: Liste des messages d'erreur si infaisable
        """
        violations = []
        schedule = self.calculate_arrival_times(route, cost_matrix, start_time)
        
        for customer, info in schedule.items():
            if customer == 0:  # Ignorer le dépôt
                continue
            
            # Vérifier si le retard dépasse la tolérance
            if info['lateness'] > self.time_tolerance:
                tw_start, tw_end = self.time_windows.get(customer, (0, 24))
                violations.append(
                    f"Client {customer}: Arrivee a {info['arrival_time']:.2f}h "
                    f"mais fenetre se termine a {tw_end:.2f}h "
                    f"(retard: {info['lateness']:.2f}h)"
                )
        
        return len(violations) == 0, violations
    
    def calculate_time_penalty(self, route: List[int],
                              cost_matrix: np.ndarray,
                              start_time: float = 6.0) -> float:
        """
        Calcule la pénalité totale pour violations de fenêtres temporelles
        
        Pénalité = somme(retard * late_penalty) pour tous les clients
        
        Args:
            route: Liste des clients
            cost_matrix: Matrice des distances
            start_time: Heure de départ
            
        Returns:
            Pénalité totale en FCFA
        """
        schedule = self.calculate_arrival_times(route, cost_matrix, start_time)
        total_penalty = 0.0
        
        for customer, info in schedule.items():
            if customer == 0:
                continue
            
            # Pénalité pour retard
            if info['lateness'] > 0:
                total_penalty += info['lateness'] * self.late_penalty
        
        return total_penalty
    
    def get_feasible_insertion_positions(self, route: List[int],
                                        customer: int,
                                        cost_matrix: np.ndarray,
                                        start_time: float = 6.0) -> List[int]:
        """
        Identifie toutes les positions d'insertion faisables pour un client
        
        Utile pour les opérateurs de réparation: permet de savoir où
        un client peut être inséré sans violer les contraintes temporelles
        
        Args:
            route: Route existante
            customer: Client à insérer
            cost_matrix: Matrice des distances
            start_time: Heure de départ
            
        Returns:
            Liste des positions faisables (indices dans la route)
        """
        feasible_positions = []
        
        # Essayer toutes les positions possibles (sauf position 0 qui est le dépôt)
        for pos in range(1, len(route)):
            # Créer une route temporaire avec le client inséré
            temp_route = route[:pos] + [customer] + route[pos:]
            
            # Vérifier la faisabilité
            is_ok, _ = self.is_feasible(temp_route, cost_matrix, start_time)
            
            if is_ok:
                feasible_positions.append(pos)
        
        return feasible_positions
    
    def get_time_window_slack(self, customer: int, 
                              arrival_time: float) -> float:
        """
        Calcule la marge temporelle (slack) pour un client
        
        Slack = temps restant avant la fin de la fenêtre
        Un slack élevé indique une flexibilité importante
        
        Args:
            customer: ID du client
            arrival_time: Temps d'arrivée prévu
            
        Returns:
            Marge temporelle en heures
        """
        tw_start, tw_end = self.time_windows.get(customer, (0, 24))
        
        # Si arrivée avant début de fenêtre
        if arrival_time < tw_start:
            return tw_end - tw_start
        
        # Si arrivée dans la fenêtre
        if arrival_time <= tw_end:
            return tw_end - arrival_time
        
        # Si arrivée après la fenêtre (négatif = retard)
        return tw_end - arrival_time
    
    def optimize_departure_times(self, route: List[int],
                                 cost_matrix: np.ndarray) -> float:
        """
        Optimise les temps de départ pour minimiser les temps d'attente
        
        Stratégie: partir le plus tard possible du dépôt tout en respectant
        toutes les fenêtres temporelles
        
        Args:
            route: Route à optimiser
            cost_matrix: Matrice des distances
            
        Returns:
            Meilleur temps de départ du dépôt
        """
        # Commencer avec un départ très tardif et ajuster
        latest_departure = 24.0  # Minuit
        
        # Calculer le temps minimum nécessaire pour la route
        total_travel_time = 0.0
        for i in range(len(route) - 1):
            distance = cost_matrix[route[i], route[i+1]]
            total_travel_time += distance / self.average_speed
        
        total_service_time = (len(route) - 2) * self.service_time
        
        # Trouver le dernier client avec la fenêtre la plus contraignante
        for i in range(1, len(route) - 1):
            customer = route[i]
            tw_start, tw_end = self.time_windows.get(customer, (0, 24))
            
            # Calculer le temps d'arrivée pour ce client
            time_to_customer = 0.0
            for j in range(i):
                distance = cost_matrix[route[j], route[j+1]]
                time_to_customer += distance / self.average_speed
                if j > 0:  # Ajouter temps de service (sauf pour le dépôt)
                    time_to_customer += self.service_time
            
            # Le départ ne peut pas être après tw_end - time_to_customer
            max_departure = tw_end - time_to_customer
            latest_departure = min(latest_departure, max_departure)
        
        # S'assurer que le départ n'est pas trop tôt (avant 6h par exemple)
        earliest_allowed = self.config.get('earliest_departure', 6.0)
        optimal_departure = max(earliest_allowed, latest_departure)
        
        return optimal_departure