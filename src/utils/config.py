"""
Configuration globale du projet VRP

Ce module centralise tous les paramètres configurables du système.
Permet de modifier facilement les paramètres sans toucher au code.
"""

class Config:
    """
    Classe de configuration pour l'algorithme VRP
    
    Attributes:
        Véhicules:
            max_vehicles: Nombre maximum de véhicules disponibles
            
        ALNS:
            max_iterations: Nombre maximum d'itérations
            initial_temperature: Température initiale pour Simulated Annealing
            cooling_rate: Taux de refroidissement (alpha)
            segment_size: Taille du segment pour mise à jour des poids
            lambda_param: Paramètre de lissage pour adaptation des poids
            
        Coûts:
            fuel_cost_per_km: Coût du carburant par kilomètre (FCFA)
            fixed_vehicle_cost: Coût fixe par véhicule (FCFA)
            average_speed: Vitesse moyenne (km/h)
            service_time: Temps de service par client (heures)
            
        Pénalités:
            overload_penalty: Pénalité pour surcharge (FCFA par unité)
            late_penalty: Pénalité pour retard (FCFA par heure)
            unassigned_penalty: Pénalité pour client non desservi (FCFA)
            vehicle_incompatibility_penalty: Pénalité pour véhicule inadapté (FCFA)
            time_restriction_penalty: Pénalité pour violation horaire (FCFA)
            
        Autres:
            seed: Graine aléatoire pour reproductibilité
            time_tolerance: Tolérance pour fenêtres temporelles (heures)
            earliest_departure: Heure de départ minimale (heures)
            start_time: Heure de début par défaut (heures)
    """
    
    def __init__(self, **kwargs):
        """
        Initialise la configuration avec les valeurs par défaut
        
        Les valeurs peuvent être surchargées via kwargs
        
        Args:
            **kwargs: Paramètres à surcharger
        """
        # ===================================================================
        # PARAMETRES DES VEHICULES
        # ===================================================================
        self.max_vehicles = kwargs.get('max_vehicles', 20)
        
        # ===================================================================
        # PARAMETRES ALNS
        # ===================================================================
        self.max_iterations = kwargs.get('max_iterations', 5000)
        self.initial_temperature = kwargs.get('initial_temperature', 1000.0)
        self.cooling_rate = kwargs.get('cooling_rate', 0.995)
        self.segment_size = kwargs.get('segment_size', 100)
        self.lambda_param = kwargs.get('lambda', 0.5)  # Paramètre de lissage
        
        # ===================================================================
        # PARAMETRES DE COUTS
        # ===================================================================
        self.fuel_cost_per_km = kwargs.get('fuel_cost_per_km', 300.0)  # FCFA/km
        self.fixed_vehicle_cost = kwargs.get('fixed_vehicle_cost', 5000.0)  # FCFA
        self.average_speed = kwargs.get('average_speed', 25.0)  # km/h
        self.service_time = kwargs.get('service_time', 0.5)  # 30 minutes en heures
        
        # ===================================================================
        # PENALITES
        # ===================================================================
        self.overload_penalty = kwargs.get('overload_penalty', 50000.0)  # FCFA par unité
        self.late_penalty = kwargs.get('late_penalty', 10000.0)  # FCFA par heure
        self.unassigned_penalty = kwargs.get('unassigned_penalty', 50000.0)  # FCFA par client
        self.vehicle_incompatibility_penalty = kwargs.get('vehicle_incompatibility_penalty', 10000.0)
        self.time_restriction_penalty = kwargs.get('time_restriction_penalty', 15000.0)
        
        # ===================================================================
        # PARAMETRES TEMPORELS
        # ===================================================================
        self.time_tolerance = kwargs.get('time_tolerance', 1.0)  # 1 heure de tolérance
        self.earliest_departure = kwargs.get('earliest_departure', 6.0)  # 6h du matin
        self.start_time = kwargs.get('start_time', 6.0)  # Heure de début par défaut
        
        # ===================================================================
        # AUTRES PARAMETRES
        # ===================================================================
        self.seed = kwargs.get('seed', 42)  # Graine aléatoire
        
    def to_dict(self):
        """
        Convertit la configuration en dictionnaire
        
        Returns:
            Dictionnaire contenant tous les paramètres
        """
        return {
            # Véhicules
            'max_vehicles': self.max_vehicles,
            
            # ALNS
            'max_iterations': self.max_iterations,
            'initial_temperature': self.initial_temperature,
            'cooling_rate': self.cooling_rate,
            'segment_size': self.segment_size,
            'lambda': self.lambda_param,
            
            # Coûts
            'fuel_cost_per_km': self.fuel_cost_per_km,
            'fixed_vehicle_cost': self.fixed_vehicle_cost,
            'average_speed': self.average_speed,
            'service_time': self.service_time,
            
            # Pénalités
            'overload_penalty': self.overload_penalty,
            'late_penalty': self.late_penalty,
            'unassigned_penalty': self.unassigned_penalty,
            'vehicle_incompatibility_penalty': self.vehicle_incompatibility_penalty,
            'time_restriction_penalty': self.time_restriction_penalty,
            
            # Temporel
            'time_tolerance': self.time_tolerance,
            'earliest_departure': self.earliest_departure,
            'start_time': self.start_time,
            
            # Autres
            'seed': self.seed
        }
    
    def __repr__(self):
        """Représentation textuelle de la configuration"""
        return (
            f"Config(\n"
            f"  Vehicules: max={self.max_vehicles}\n"
            f"  ALNS: iterations={self.max_iterations}, T0={self.initial_temperature}, "
            f"alpha={self.cooling_rate}\n"
            f"  Couts: carburant={self.fuel_cost_per_km} FCFA/km, "
            f"vitesse={self.average_speed} km/h\n"
            f"  Penalites: surcharge={self.overload_penalty}, "
            f"retard={self.late_penalty}, non_dessservi={self.unassigned_penalty}\n"
            f"  Seed: {self.seed}\n"
            f")"
        )
    
    def print_config(self):
        """Affiche la configuration de manière formatée"""
        print("\n" + "="*70)
        print("CONFIGURATION DU SYSTEME VRP")
        print("="*70)
        
        print("\n[VEHICULES]")
        print(f"  Nombre maximum: {self.max_vehicles}")
        
        print("\n[ALNS - PARAMETRES ALGORITHME]")
        print(f"  Iterations maximum: {self.max_iterations}")
        print(f"  Temperature initiale: {self.initial_temperature}")
        print(f"  Taux de refroidissement: {self.cooling_rate}")
        print(f"  Taille segment: {self.segment_size}")
        print(f"  Parametre lambda: {self.lambda_param}")
        
        print("\n[COUTS ET VITESSE]")
        print(f"  Cout carburant: {self.fuel_cost_per_km} FCFA/km")
        print(f"  Cout fixe vehicule: {self.fixed_vehicle_cost} FCFA")
        print(f"  Vitesse moyenne: {self.average_speed} km/h")
        print(f"  Temps de service: {self.service_time*60:.0f} minutes")
        
        print("\n[PENALITES]")
        print(f"  Surcharge: {self.overload_penalty} FCFA/unite")
        print(f"  Retard: {self.late_penalty} FCFA/heure")
        print(f"  Client non desservi: {self.unassigned_penalty} FCFA")
        print(f"  Vehicule inadapte: {self.vehicle_incompatibility_penalty} FCFA")
        print(f"  Violation horaire: {self.time_restriction_penalty} FCFA")
        
        print("\n[CONTRAINTES TEMPORELLES]")
        print(f"  Tolerance fenetre: {self.time_tolerance} heure(s)")
        print(f"  Depart minimal: {self.earliest_departure}h")
        print(f"  Heure de debut: {self.start_time}h")
        
        print("\n[REPRODUCTIBILITE]")
        print(f"  Seed aleatoire: {self.seed}")
        
        print("="*70 + "\n")
    
    @classmethod
    def from_file(cls, filepath: str):
        """
        Charge la configuration depuis un fichier JSON
        
        Args:
            filepath: Chemin vers le fichier JSON
            
        Returns:
            Instance de Config
        """
        import json
        
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        return cls(**params)
    
    def save_to_file(self, filepath: str):
        """
        Sauvegarde la configuration dans un fichier JSON
        
        Args:
            filepath: Chemin de destination
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"Configuration sauvegardee dans: {filepath}")


# Configuration par défaut (peut être importée directement)
DEFAULT_CONFIG = Config()


def get_default_config():
    """
    Retourne une configuration par défaut
    
    Returns:
        Instance de Config avec valeurs par défaut
    """
    return Config()


def get_fast_config():
    """
    Configuration pour tests rapides (peu d'itérations)
    
    Returns:
        Instance de Config optimisée pour la vitesse
    """
    return Config(
        max_iterations=1000,
        initial_temperature=500,
        cooling_rate=0.99,
        segment_size=50
    )


def get_quality_config():
    """
    Configuration pour recherche de qualité (beaucoup d'itérations)
    
    Returns:
        Instance de Config optimisée pour la qualité
    """
    return Config(
        max_iterations=10000,
        initial_temperature=2000,
        cooling_rate=0.998,
        segment_size=200
    )


if __name__ == '__main__':
    # Test de la configuration
    config = Config()
    config.print_config()
    
    # Test de sauvegarde/chargement
    config.save_to_file('config_test.json')
    config_loaded = Config.from_file('config_test.json')
    print("\nConfiguration chargee depuis fichier:")
    print(config_loaded)