"""
Gestion des restrictions spécifiques de Yaoundé

Contraintes particulières au contexte:
- Restrictions horaires (6h-10h pour certaines zones)
- Zones présidentielles (Etoudi)
- Conditions de stationnement
- Praticabilité selon les conditions météo
- Compatibilité véhicule-route
"""

from typing import Dict, List, Tuple, Set

class RestrictionManager:
    """
    Gestionnaire des restrictions contextuelles de Yaoundé
    
    Gère les contraintes spécifiques qui ne sont ni temporelles ni de capacité,
    mais liées au contexte local de la ville.
    """
    
    def __init__(self, route_data: Dict, config: Dict):
        """
        Initialisation du gestionnaire
        
        Args:
            route_data: Données des trajets (stationnement, restrictions, etc.)
            config: Configuration générale
        """
        self.route_data = route_data
        self.config = config
        
        # Extraire les restrictions depuis les données
        self._parse_restrictions()
    
    def _parse_restrictions(self):
        """
        Parse les données pour extraire les différentes restrictions
        
        Crée des dictionnaires pour accès rapide:
        - restricted_hours: clients avec restrictions horaires
        - presidential_zones: zones avec restrictions spéciales
        - parking_difficulty: difficulté de stationnement par client
        - road_conditions: état des routes
        - vehicle_requirements: type de véhicule requis par route
        """
        self.restricted_hours = {}  # {client_id: (debut, fin, description)}
        self.presidential_zones = set()  # Ensemble des zones présidentielles
        self.parking_difficulty = {}  # {client_id: niveau_difficulte}
        self.road_conditions = {}  # {(from, to): etat_route}
        self.vehicle_requirements = {}  # {client_id: type_vehicule_requis}
        
        # Parser les données (exemple de structure)
        # Dans la pratique, ceci serait adapté à la structure réelle du CSV
        for client_id, data in self.route_data.items():
            # Restrictions horaires
            if 'Restrictions Horaires' in data and data['Restrictions Horaires'] != 'Aucune':
                restriction = data['Restrictions Horaires']
                self.restricted_hours[client_id] = self._parse_time_restriction(restriction)
            
            # Zones présidentielles
            if 'zone présidentielle' in str(data.get('Restrictions Horaires', '')).lower():
                self.presidential_zones.add(client_id)
            
            # Stationnement
            if 'Stationnement' in data:
                self.parking_difficulty[client_id] = data['Stationnement']
            
            # Type de véhicule requis
            if 'Type de Véhicule' in data:
                required = data['Type de Véhicule']
                if required not in ['Tous', 'tous']:
                    self.vehicle_requirements[client_id] = required
    
    def _parse_time_restriction(self, restriction_str: str) -> Tuple:
        """
        Parse une chaîne de restriction horaire
        
        Exemples:
        - "6h-10h seulement" -> (6, 10, "seulement")
        - "Eviter nuit" -> (6, 20, "eviter_nuit")
        
        Args:
            restriction_str: Chaîne décrivant la restriction
            
        Returns:
            (heure_debut, heure_fin, type_restriction)
        """
        restriction_lower = restriction_str.lower()
        
        # Pattern "Xh-Yh seulement"
        if 'h-' in restriction_lower and 'seulement' in restriction_lower:
            parts = restriction_lower.split('seulement')[0].strip()
            hours = parts.replace('h', '').split('-')
            start = int(hours[0])
            end = int(hours[1])
            return (start, end, "only")
        
        # Pattern "Eviter nuit"
        if 'nuit' in restriction_lower:
            return (6, 20, "avoid_night")
        
        # Pattern "Jours matchs"
        if 'match' in restriction_lower:
            return (0, 24, "match_days")
        
        # Par défaut, pas de restriction
        return (0, 24, "none")
    
    def is_time_allowed(self, customer: int, arrival_time: float) -> Tuple[bool, str]:
        """
        Vérifie si l'arrivée à un client est permise à une heure donnée
        
        Args:
            customer: ID du client
            arrival_time: Heure d'arrivée (format 24h, ex: 14.5 pour 14h30)
            
        Returns:
            (autorise, raison_si_refuse)
        """
        if customer not in self.restricted_hours:
            return True, ""
        
        start, end, restriction_type = self.restricted_hours[customer]
        
        if restriction_type == "only":
            # Accès autorisé uniquement pendant [start, end]
            if start <= arrival_time <= end:
                return True, ""
            else:
                return False, f"Acces autorise uniquement entre {start}h et {end}h"
        
        elif restriction_type == "avoid_night":
            # Eviter la nuit (6h-20h recommandé)
            if start <= arrival_time <= end:
                return True, ""
            else:
                return False, f"Eviter les heures de nuit (recommande: {start}h-{end}h)"
        
        return True, ""
    
    def is_vehicle_compatible(self, vehicle_type: str, customer: int) -> Tuple[bool, str]:
        """
        Vérifie si un type de véhicule peut desservir un client
        
        Certaines routes nécessitent des véhicules spécifiques:
        - Routes en terre/latérite: 4x4 recommandé
        - Routes en mauvais état: 4x4 requis
        
        Args:
            vehicle_type: Type du véhicule (ex: "Camionnette", "4x4")
            customer: ID du client
            
        Returns:
            (compatible, raison_si_incompatible)
        """
        if customer not in self.vehicle_requirements:
            return True, ""
        
        required = self.vehicle_requirements[customer]
        
        # Si "recommandé" mais pas obligatoire
        if "recommandé" in required.lower() or "recommande" in required.lower():
            if vehicle_type != required.split()[0]:
                return True, f"Vehicule {required} recommande mais pas obligatoire"
            return True, ""
        
        # Si obligatoire (ex: "4x4")
        if required.lower() in vehicle_type.lower():
            return True, ""
        
        return False, f"Vehicule requis: {required}"
    
    def get_parking_penalty(self, customer: int) -> float:
        """
        Calcule la pénalité liée aux difficultés de stationnement
        
        Niveaux de difficulté:
        - "Excellent": 0 FCFA
        - "Moyen": 1000 FCFA
        - "Difficile": 2500 FCFA
        - "Très difficile": 5000 FCFA
        - "Limité" / "Très limité": 3000 FCFA
        
        Args:
            customer: ID du client
            
        Returns:
            Pénalité en FCFA
        """
        if customer not in self.parking_difficulty:
            return 0.0
        
        difficulty = self.parking_difficulty[customer].lower()
        
        if difficulty == "excellent":
            return 0.0
        elif difficulty == "moyen":
            return 1000.0
        elif difficulty == "difficile":
            return 2500.0
        elif "très difficile" in difficulty or "tres difficile" in difficulty:
            return 5000.0
        elif "limité" in difficulty or "limite" in difficulty:
            return 3000.0
        
        return 0.0
    
    def is_route_feasible_with_restrictions(self, route_customers: List[int],
                                           vehicle_type: str,
                                           arrival_times: Dict[int, float]) -> Tuple[bool, List[str]]:
        """
        Vérifie si une route respecte toutes les restrictions contextuelles
        
        Args:
            route_customers: Liste des clients de la route
            vehicle_type: Type de véhicule utilisé
            arrival_times: Dictionnaire {client: temps_arrivee}
            
        Returns:
            (faisabilite, liste_violations)
        """
        violations = []
        
        for customer in route_customers:
            if customer == 0:  # Ignorer le dépôt
                continue
            
            # Vérifier compatibilité véhicule
            vehicle_ok, vehicle_msg = self.is_vehicle_compatible(vehicle_type, customer)
            if not vehicle_ok:
                violations.append(f"Client {customer}: {vehicle_msg}")
            
            # Vérifier restrictions horaires
            if customer in arrival_times:
                time_ok, time_msg = self.is_time_allowed(customer, arrival_times[customer])
                if not time_ok:
                    violations.append(f"Client {customer}: {time_msg}")
        
        return len(violations) == 0, violations
    
    def calculate_restriction_penalties(self, route_customers: List[int],
                                       vehicle_type: str,
                                       arrival_times: Dict[int, float]) -> float:
        """
        Calcule les pénalités totales liées aux restrictions
        
        Inclut:
        - Pénalités de stationnement
        - Pénalités pour véhicule non adapté
        - Pénalités pour accès hors horaires recommandés
        
        Args:
            route_customers: Liste des clients de la route
            vehicle_type: Type de véhicule utilisé
            arrival_times: Dictionnaire {client: temps_arrivee}
            
        Returns:
            Pénalité totale en FCFA
        """
        total_penalty = 0.0
        
        for customer in route_customers:
            if customer == 0:
                continue
            
            # Pénalité de stationnement
            total_penalty += self.get_parking_penalty(customer)
            
            # Pénalité pour véhicule non adapté
            vehicle_ok, _ = self.is_vehicle_compatible(vehicle_type, customer)
            if not vehicle_ok:
                total_penalty += self.config.get('vehicle_incompatibility_penalty', 10000)
            
            # Pénalité pour accès hors horaires
            if customer in arrival_times:
                time_ok, _ = self.is_time_allowed(customer, arrival_times[customer])
                if not time_ok:
                    total_penalty += self.config.get('time_restriction_penalty', 15000)
        
        return total_penalty
    
    def get_presidential_zone_restrictions(self) -> Set[int]:
        """
        Retourne l'ensemble des clients dans des zones présidentielles
        
        Ces zones nécessitent une attention particulière et des autorisations
        
        Returns:
            Ensemble des IDs de clients en zone présidentielle
        """
        return self.presidential_zones
    
    def is_rainy_season_feasible(self, route_customers: List[int],
                                 vehicle_type: str,
                                 praticability_data: Dict) -> Tuple[bool, List[str]]:
        """
        Vérifie la faisabilité d'une route en saison des pluies
        
        Certaines routes deviennent impraticables ou nécessitent des véhicules 4x4
        
        Args:
            route_customers: Liste des clients
            vehicle_type: Type de véhicule
            praticability_data: Données de praticabilité {client: niveau}
            
        Returns:
            (faisabilite, liste_avertissements)
        """
        warnings = []
        
        for customer in route_customers:
            if customer == 0:
                continue
            
            praticability = praticability_data.get(customer, "Bonne")
            
            # Routes très mauvaises nécessitent absolument un 4x4
            if praticability in ["Très mauvaise", "Tres mauvaise", "Mauvaise"]:
                if "4x4" not in vehicle_type:
                    warnings.append(
                        f"Client {customer}: Route {praticability} en saison des pluies, "
                        f"4x4 fortement recommande (vehicule actuel: {vehicle_type})"
                    )
        
        # La route est considérée infaisable s'il y a des avertissements critiques
        critical_warnings = [w for w in warnings if "Tres mauvaise" in w or "Très mauvaise" in w]
        
        return len(critical_warnings) == 0, warnings
    
    def get_security_level(self, customer: int) -> str:
        """
        Retourne le niveau de sécurité d'une zone
        
        Args:
            customer: ID du client
            
        Returns:
            Niveau de sécurité ("Elevée", "Moyenne", "Faible")
        """
        if customer in self.route_data and 'Sécurité' in self.route_data[customer]:
            return self.route_data[customer]['Sécurité']
        return "Moyenne"  # Par défaut
    
    def recommend_route_adjustments(self, route_customers: List[int],
                                   vehicle_type: str) -> List[str]:
        """
        Propose des ajustements pour améliorer une route
        
        Args:
            route_customers: Liste des clients de la route
            vehicle_type: Type de véhicule utilisé
            
        Returns:
            Liste de recommandations
        """
        recommendations = []
        
        # Vérifier les zones à faible sécurité
        low_security = [c for c in route_customers 
                       if self.get_security_level(c) == "Faible"]
        if low_security:
            recommendations.append(
                f"Attention: {len(low_security)} client(s) en zone de securite faible. "
                f"Prevoir accompagnement ou livraison groupee."
            )
        
        # Vérifier le stationnement difficile
        difficult_parking = [c for c in route_customers 
                           if "difficile" in self.parking_difficulty.get(c, "").lower()]
        if len(difficult_parking) > 3:
            recommendations.append(
                f"{len(difficult_parking)} clients avec stationnement difficile. "
                f"Prevoir temps supplementaire."
            )
        
        # Vérifier la compatibilité véhicule
        incompatible = []
        for customer in route_customers:
            ok, _ = self.is_vehicle_compatible(vehicle_type, customer)
            if not ok:
                incompatible.append(customer)
        
        if incompatible:
            recommendations.append(
                f"Vehicule {vehicle_type} non optimal pour {len(incompatible)} client(s). "
                f"Envisager un 4x4."
            )
        
        return recommendations