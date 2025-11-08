"""
Adaptive Large Neighborhood Search (ALNS) pour VRP
Implémentation complète de l'algorithme d'optimisation

L'ALNS fonctionne en itérant les étapes suivantes:
1. Détruire partiellement la solution courante (retirer des clients)
2. Réparer la solution (réinsérer les clients)
3. Accepter ou rejeter la nouvelle solution (Simulated Annealing)
4. Adapter les poids des opérateurs selon leur performance

Référence: Ropke & Pisinger (2006) - "An Adaptive Large Neighborhood Search Heuristic 
for the Pickup and Delivery Problem with Time Windows"
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import copy

from src.models.solution import Solution
from src.algorithms.destroy_operators import DestroyOperators
from src.algorithms.repair_operators import RepairOperators
from src.algorithms.local_search import LocalSearch
from src.algorithms.initial_solution import InitialSolutionGenerator

class ALNS:
    """
    Classe principale implémentant l'algorithme ALNS
    
    Attributes:
        cost_matrix: Matrice des coûts entre chaque paire de nœuds
        demands: Dictionnaire {client_id: demande}
        time_windows: Dictionnaire {client_id: (début, fin)}
        vehicle_capacities: Dictionnaire {type_véhicule: capacité}
        config: Configuration générale de l'algorithme
        
        destroy_ops: Opérateurs de destruction
        repair_ops: Opérateurs de réparation
        local_search: Opérateurs de recherche locale
        
        destroy_weights: Poids actuels des opérateurs de destruction
        repair_weights: Poids actuels des opérateurs de réparation
        destroy_scores: Scores cumulés des opérateurs de destruction
        repair_scores: Scores cumulés des opérateurs de réparation
        
        temperature: Température courante pour Simulated Annealing
        alpha: Taux de refroidissement
    """
    
    def __init__(self, cost_matrix: np.ndarray,
                 demands: Dict[int, float],
                 time_windows: Dict[int, Tuple[float, float]],
                 vehicle_capacities: Dict[str, float],
                 config: Dict):
        
        # Données du problème
        self.cost_matrix = cost_matrix
        self.demands = demands
        self.time_windows = time_windows
        self.vehicle_capacities = vehicle_capacities
        self.config = config
        
        # Initialiser les opérateurs de destruction, réparation et recherche locale
        self.destroy_ops = DestroyOperators(config)
        self.repair_ops = RepairOperators(
            cost_matrix, demands, time_windows, 
            vehicle_capacities, config
        )
        self.local_search = LocalSearch(cost_matrix, config)
        
        # Initialisation des poids des opérateurs
        # Au départ, tous les opérateurs ont le même poids (équiprobables)
        self.destroy_weights = np.ones(5)  # 5 opérateurs de destruction
        self.repair_weights = np.ones(4)   # 4 opérateurs de réparation
        
        # Scores pour tracking de la performance
        # Ces scores seront mis à jour à chaque utilisation d'un opérateur
        self.destroy_scores = np.zeros(5)
        self.repair_scores = np.zeros(4)
        self.destroy_usage = np.zeros(5)  # Nombre d'utilisations
        self.repair_usage = np.zeros(4)
        
        # Paramètres du Simulated Annealing
        self.temperature = config.get('initial_temperature', 1000)
        self.alpha = config.get('cooling_rate', 0.995)  # Facteur de refroidissement
        
        # Paramètres de l'adaptation des poids
        self.segment_size = config.get('segment_size', 100)  # Mise à jour tous les N iter
        self.lambda_param = config.get('lambda', 0.5)  # Paramètre de lissage
        
        # Système de récompenses (scores attribués selon le résultat)
        self.score_best = 13      # Nouvelle meilleure solution globale
        self.score_better = 9     # Solution meilleure que la courante
        self.score_accepted = 3   # Solution acceptée mais pas meilleure
        self.score_rejected = 0   # Solution rejetée
        
        # Historique pour analyse post-exécution
        self.history = {
            'best_costs': [],        # Evolution du meilleur coût
            'current_costs': [],     # Evolution du coût courant
            'temperatures': [],      # Evolution de la température
            'acceptance_rate': []    # Taux d'acceptation par segment
        }
    
    def solve(self, max_iterations: int = 5000) -> Tuple[Solution, Dict]:
        """
        Exécute l'algorithme ALNS pour résoudre le VRP
        
        Processus:
        1. Génération de la solution initiale
        2. Boucle principale ALNS:
           - Destruction partielle
           - Réparation
           - Critère d'acceptation
           - Mise à jour adaptative des poids
        3. Post-optimisation par recherche locale
        
        Args:
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            (meilleure_solution, statistiques_execution)
        """
        print("\n" + "="*60)
        print("DEMARRAGE ALNS - Vehicle Routing Problem Yaounde")
        print("="*60)
        
        start_time = time.time()
        
        # ==================================================================
        # ETAPE 1: GENERATION DE LA SOLUTION INITIALE
        # ==================================================================
        print("\n[PHASE 1] Generation de la solution initiale...")
        print("-" * 60)
        
        initial_gen = InitialSolutionGenerator(
            self.cost_matrix, self.demands, self.time_windows,
            self.vehicle_capacities, self.config
        )
        current_solution = initial_gen.generate_nearest_neighbor()
        best_solution = current_solution.copy()
        
        print(f"Solution initiale generee:")
        print(f"  - Cout total: {current_solution.total_cost:,.0f} FCFA")
        print(f"  - Nombre de vehicules: {current_solution.num_vehicles_used}")
        print(f"  - Distance totale: {current_solution.total_distance:.1f} km")
        print(f"  - Clients desservis: {len(current_solution.get_all_customers())}")
        
        # Variables de suivi
        iterations_since_improvement = 0  # Pour détecter la stagnation
        accepted_solutions = 0            # Pour calculer le taux d'acceptation
        
        print(f"\n[PHASE 2] Execution de {max_iterations} iterations ALNS")
        print("-" * 60)
        
        # ==================================================================
        # ETAPE 2: BOUCLE PRINCIPALE ALNS
        # ==================================================================
        for iteration in range(max_iterations):
            
            # --------------------------------------------------------------
            # 2.1 SELECTION DES OPERATEURS (Roulette Wheel Selection)
            # --------------------------------------------------------------
            # Les opérateurs avec des poids plus élevés ont plus de chances
            # d'être sélectionnés
            destroy_idx = self._select_operator(self.destroy_weights)
            repair_idx = self._select_operator(self.repair_weights)
            
            # --------------------------------------------------------------
            # 2.2 PHASE DE DESTRUCTION
            # --------------------------------------------------------------
            # Calculer le nombre de clients à retirer (entre 10% et 40%)
            q = self._calculate_destroy_size(current_solution)
            
            # Appliquer l'opérateur de destruction sélectionné
            destroyed_solution = self._apply_destroy(
                current_solution, destroy_idx, q
            )
            
            # --------------------------------------------------------------
            # 2.3 PHASE DE REPARATION
            # --------------------------------------------------------------
            # Réinsérer les clients retirés dans la solution
            new_solution = self._apply_repair(destroyed_solution, repair_idx)
            
            # Recalculer toutes les métriques de la nouvelle solution
            new_solution.calculate_metrics(
                self.cost_matrix, self.demands, self.config
            )
            
            # --------------------------------------------------------------
            # 2.4 CRITERE D'ACCEPTATION (Simulated Annealing)
            # --------------------------------------------------------------
            # Calculer la différence de coût
            delta = new_solution.total_cost - current_solution.total_cost
            accept = False
            score = self.score_rejected
            
            if delta < 0:
                # Solution strictement meilleure: toujours accepter
                accept = True
                
                if new_solution.total_cost < best_solution.total_cost:
                    # Nouvelle meilleure solution globale
                    best_solution = new_solution.copy()
                    score = self.score_best
                    iterations_since_improvement = 0
                    
                    print(f"\n[Iteration {iteration}] NOUVELLE MEILLEURE SOLUTION")
                    print(f"  Cout: {best_solution.total_cost:,.0f} FCFA "
                          f"(amelioration: {-delta:,.0f} FCFA)")
                else:
                    # Solution meilleure que la courante mais pas que la best
                    score = self.score_better
                    
            elif np.random.random() < np.exp(-delta / self.temperature):
                # Acceptation probabiliste d'une solution moins bonne
                # Probabilité = exp(-delta/T)
                # Plus T est élevé, plus on accepte facilement les solutions pires
                accept = True
                score = self.score_accepted
            
            # Mettre à jour la solution courante si acceptée
            if accept:
                current_solution = new_solution
                accepted_solutions += 1
            
            # --------------------------------------------------------------
            # 2.5 MISE A JOUR DES SCORES
            # --------------------------------------------------------------
            # Attribuer le score aux opérateurs utilisés
            self.destroy_scores[destroy_idx] += score
            self.repair_scores[repair_idx] += score
            self.destroy_usage[destroy_idx] += 1
            self.repair_usage[repair_idx] += 1
            
            # --------------------------------------------------------------
            # 2.6 MISE A JOUR ADAPTATIVE DES POIDS
            # --------------------------------------------------------------
            # Tous les 'segment_size' itérations, recalculer les poids
            if (iteration + 1) % self.segment_size == 0:
                self._update_weights()
                
                # Calculer et afficher les statistiques du segment
                acceptance_rate = accepted_solutions / self.segment_size
                segment_num = (iteration + 1) // self.segment_size
                
                print(f"\n[Segment {segment_num}] Iteration {iteration+1}/{max_iterations}")
                print(f"  Meilleur cout: {best_solution.total_cost:,.0f} FCFA")
                print(f"  Cout courant: {current_solution.total_cost:,.0f} FCFA")
                print(f"  Temperature: {self.temperature:.1f}")
                print(f"  Taux d'acceptation: {acceptance_rate:.1%}")
                print(f"  Iterations sans amelioration: {iterations_since_improvement}")
                
                # Réinitialiser le compteur d'acceptations
                accepted_solutions = 0
            
            # --------------------------------------------------------------
            # 2.7 REFROIDISSEMENT (Cooling Schedule)
            # --------------------------------------------------------------
            self.temperature *= self.alpha
            iterations_since_improvement += 1
            
            # --------------------------------------------------------------
            # 2.8 MECANISME DE RECHAUFFAGE (Reheating)
            # --------------------------------------------------------------
            # Si l'algorithme stagne (pas d'amélioration depuis longtemps),
            # augmenter la température pour explorer davantage
            if iterations_since_improvement > 500:
                self.temperature *= 2
                iterations_since_improvement = 0
                print(f"\n[RECHAUFFAGE] Temperature augmentee a {self.temperature:.1f}")
            
            # Enregistrer l'historique pour analyse
            self.history['best_costs'].append(best_solution.total_cost)
            self.history['current_costs'].append(current_solution.total_cost)
            self.history['temperatures'].append(self.temperature)
        
        # ==================================================================
        # ETAPE 3: POST-OPTIMISATION PAR RECHERCHE LOCALE
        # ==================================================================
        print("\n[PHASE 3] Post-optimisation avec recherche locale (2-opt)")
        print("-" * 60)
        
        initial_cost = best_solution.total_cost
        best_solution = self.local_search.two_opt(best_solution)
        best_solution.calculate_metrics(
            self.cost_matrix, self.demands, self.config
        )
        improvement = initial_cost - best_solution.total_cost
        
        if improvement > 0:
            print(f"Amelioration par 2-opt: {improvement:,.0f} FCFA")
        
        elapsed_time = time.time() - start_time
        
        # ==================================================================
        # AFFICHAGE DES RESULTATS FINAUX
        # ==================================================================
        print("\n" + "="*60)
        print("ALNS TERMINE - RESULTATS FINAUX")
        print("="*60)
        print(f"Temps d'execution: {elapsed_time:.2f} secondes")
        print(f"\nSolution optimale:")
        print(f"  - Cout total: {best_solution.total_cost:,.0f} FCFA")
        print(f"  - Nombre de vehicules: {best_solution.num_vehicles_used}")
        print(f"  - Distance totale: {best_solution.total_distance:.1f} km")
        print(f"  - Clients desservis: {len(best_solution.get_all_customers())}")
        
        if best_solution.unassigned_customers:
            print(f"  - ATTENTION: {len(best_solution.unassigned_customers)} "
                  f"clients non desservis")
        
        # Calculer l'amélioration par rapport à la solution initiale
        initial_cost = self.history['best_costs'][0]
        improvement_pct = ((initial_cost - best_solution.total_cost) / 
                          initial_cost * 100)
        print(f"\nAmelioration totale: {improvement_pct:.2f}%")
        
        # ==================================================================
        # CONSTRUCTION DES STATISTIQUES
        # ==================================================================
        stats = {
            'execution_time': elapsed_time,
            'iterations': max_iterations,
            'final_cost': best_solution.total_cost,
            'initial_cost': initial_cost,
            'improvement_percent': improvement_pct,
            'num_vehicles': best_solution.num_vehicles_used,
            'total_distance': best_solution.total_distance,
            'history': self.history,
            'operator_usage': {
                'destroy': self.destroy_usage.tolist(),
                'repair': self.repair_usage.tolist()
            },
            'operator_performance': {
                'destroy_avg_scores': (self.destroy_scores / 
                                      np.maximum(self.destroy_usage, 1)).tolist(),
                'repair_avg_scores': (self.repair_scores / 
                                     np.maximum(self.repair_usage, 1)).tolist()
            }
        }
        
        return best_solution, stats
    
    def _select_operator(self, weights: np.ndarray) -> int:
        """
        Sélection d'un opérateur par méthode de la roulette (Roulette Wheel Selection)
        
        Les opérateurs avec des poids plus élevés ont plus de chances d'être choisis.
        
        Args:
            weights: Tableau des poids des opérateurs
            
        Returns:
            Index de l'opérateur sélectionné
        """
        # Normaliser les poids pour obtenir des probabilités
        probabilities = weights / weights.sum()
        
        # Sélection aléatoire selon les probabilités
        return np.random.choice(len(weights), p=probabilities)
    
    def _calculate_destroy_size(self, solution: Solution) -> int:
        """
        Calcule le nombre de clients à détruire
        
        Stratégie: retirer entre 10% et 40% des clients
        Un nombre trop faible ne permet pas assez de diversification
        Un nombre trop élevé rend la réparation difficile
        
        Args:
            solution: Solution courante
            
        Returns:
            Nombre de clients à retirer
        """
        total_customers = len(solution.get_all_customers())
        
        # Bornes: minimum 1 client, maximum 40% des clients
        min_destroy = max(1, int(0.1 * total_customers))
        max_destroy = max(1, int(0.4 * total_customers))
        
        # Sélection aléatoire dans l'intervalle
        return np.random.randint(min_destroy, max_destroy + 1)
    
    def _apply_destroy(self, solution: Solution, op_idx: int, q: int) -> Solution:
        """
        Applique l'opérateur de destruction sélectionné
        
        Opérateurs disponibles:
        0: Random removal - Retrait aléatoire
        1: Worst removal - Retrait des pires clients
        2: Shaw removal - Retrait de clients similaires
        3: Route removal - Retrait d'une route complète
        4: Cluster removal - Retrait d'un cluster géographique
        
        Args:
            solution: Solution à détruire
            op_idx: Index de l'opérateur
            q: Nombre de clients à retirer
            
        Returns:
            Solution partiellement détruite
        """
        if op_idx == 0:
            new_sol, _ = self.destroy_ops.random_removal(solution, q)
        elif op_idx == 1:
            new_sol, _ = self.destroy_ops.worst_removal(
                solution, q, self.cost_matrix
            )
        elif op_idx == 2:
            new_sol, _ = self.destroy_ops.shaw_removal(
                solution, q, self.cost_matrix, self.time_windows
            )
        elif op_idx == 3:
            new_sol, _ = self.destroy_ops.route_removal(solution, q)
        else:  # op_idx == 4
            new_sol, _ = self.destroy_ops.cluster_removal(
                solution, q, self.cost_matrix
            )
        
        return new_sol
    
    def _apply_repair(self, solution: Solution, op_idx: int) -> Solution:
        """
        Applique l'opérateur de réparation sélectionné
        
        Opérateurs disponibles:
        0: Greedy insertion - Insertion gloutonne
        1: Regret insertion - Insertion avec regret-3
        2: Time-oriented insertion - Insertion chronologique
        3: Random insertion - Insertion aléatoire
        
        Args:
            solution: Solution partielle à réparer
            op_idx: Index de l'opérateur
            
        Returns:
            Solution réparée (complète)
        """
        if op_idx == 0:
            return self.repair_ops.greedy_insertion(solution)
        elif op_idx == 1:
            return self.repair_ops.regret_insertion(solution, k=3)
        elif op_idx == 2:
            return self.repair_ops.time_oriented_insertion(solution)
        else:  # op_idx == 3
            return self.repair_ops.random_insertion(solution)
    
    def _update_weights(self):
        """
        Mise à jour adaptative des poids des opérateurs
        
        Formule: w_new = lambda * w_old + (1-lambda) * score_avg
        
        Où:
        - lambda est le paramètre de lissage (0.5 par défaut)
        - score_avg est le score moyen obtenu par l'opérateur
        
        Cette approche permet de:
        - Favoriser les opérateurs performants
        - Garder une part d'exploration (via lambda)
        - S'adapter dynamiquement à l'évolution du problème
        """
        # Mise à jour des poids des opérateurs de destruction
        for i in range(len(self.destroy_weights)):
            if self.destroy_usage[i] > 0:
                # Calculer le score moyen de cet opérateur
                avg_score = self.destroy_scores[i] / self.destroy_usage[i]
                
                # Formule de mise à jour avec lissage exponentiel
                self.destroy_weights[i] = (
                    self.lambda_param * self.destroy_weights[i] +
                    (1 - self.lambda_param) * avg_score
                )
        
        # Mise à jour des poids des opérateurs de réparation
        for i in range(len(self.repair_weights)):
            if self.repair_usage[i] > 0:
                avg_score = self.repair_scores[i] / self.repair_usage[i]
                self.repair_weights[i] = (
                    self.lambda_param * self.repair_weights[i] +
                    (1 - self.lambda_param) * avg_score
                )
        
        # Normalisation optionnelle pour garder les poids dans une plage raisonnable
        # (évite les débordements numériques après de nombreuses itérations)
        self.destroy_weights = np.maximum(self.destroy_weights, 0.1)
        self.repair_weights = np.maximum(self.repair_weights, 0.1)
        
        # Réinitialiser les scores et compteurs pour le prochain segment
        self.destroy_scores = np.zeros(len(self.destroy_weights))
        self.repair_scores = np.zeros(len(self.repair_weights))
        self.destroy_usage = np.zeros(len(self.destroy_weights))
        self.repair_usage = np.zeros(len(self.repair_weights))