import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.extmath import softmax

class CoutoBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    Implementação de AdaBoost baseada na
    interface pública dos classificadores
    do sk-learn.
    """
    def __init__(self, max_estimators=108):
        self.max_estimators = max_estimators

    
    def fit(self, X, y):
        """
        Treina o classificador no conjunto
        de exemplos X com rótulos y.
        
        Parâmetros
        ----------
        X : ndarray da forma (n_exemplos, n_caracteristicas)
            Uma matriz com características
            binárias para cada exemplo.
            
        y : ndarray da forma (n_exemplos,)
            Os rótulos com valores -1 ou 1.
            
        
        Retorna
        -------
        self : object
            O estimador treinado.
        """
        self.classes_ = [-1, 1]
        
        # Em alto nível:
        # 1. cria stumps (seguindo uma estratégia específica para tic-tac-toe)
        self.stumps = self._create_stumps(X)
        # print("stumps totais: ", len(self.stumps))
        
        # 2. inicializa escolhas das iterações
        self.iteration_errors = []
        self.iteration_alphas = []
        self.iteration_stumps = []
        
        # 3. cria pesos dos exemplos
        samples = X.shape[0]
        self.input_weights = np.full(samples, 1/samples)

        # 4. repete
        for t in range(self.max_estimators):
            # print("Iteração t=", t)
            # 4.1. seleciona melhor stump (e o remove da lista de disponíveis)
            error, mistakes = self._pick_best_stump(X, y)
            
            # 4.2. calcula importância do stump selecionado
            # 4.3. atualiza os pesos considerando esse stump
            self._update_weights(error, mistakes)

            # 4.4. se erro = 0 ou selecionou max_estimators, sai
            if error == 0:
                print("Saiu mais cedo porque erro empírico zerou na iteração " + str(t))
                break

        
        return self
        

    def predict(self, X, max_estimators=None):
        """
        Classifica o vetor de exemplos X.
        
        Parâmetros
        -----------
        X : ndarray da forma (n_exemplos, n_características)
            Matriz de exemplos a serem 
            classificados.
            
        Retorna
        -------
        y : ndarray da forma (n_exemplos,)
            As classes escolhidas.
        """
        classified = self.decision_function(X, max_estimators)
        classified[classified > 0] = +1
        classified[classified < 0] = -1
            
        return classified
           

    def decision_function(self, X, max_estimators=None):
        # Em alto nível
        # 1. invoca todos os stumps
        # 2. multiplica o resultado de cada um pelo alfa
        # 3. soma tudo e retorna
        iterations = np.minimum(
            (max_estimators or self.max_estimators), len(self.iteration_stumps))
        decision = np.zeros(X.shape[0])
        for i in range(X.shape[0]):

            decision[i] = 0
            for t, stump_id in enumerate(self.iteration_stumps):
                if t > iterations - 1:
                    break

                (feature, value, label) = self.stumps[stump_id]

                alpha = self.iteration_alphas[t]

                if X[i, feature] == value:
                    decision[i] += alpha * label
                else:
                    decision[i] -= alpha * label

        return decision


    def predict_proba(self, X):
        decision = self.decision_function(X)
        decision = np.vstack([-decision, decision]).T / 2
        return softmax(decision, copy=False)


    def _create_all_posible_stumps(self, X):
        stumps = []

        # cada característica da entrada (3 por casinha)
        for feature in range(X.shape[1]):
            # valor binário da feature
            for value in [0, 1]:
                # cada valor de positivo ou negativo
                for label in [-1, 1]:
                    stumps.append((feature, value, label))

        return stumps

    
    def _create_stumps(self, X):
        all_stumps = np.array(self._create_all_posible_stumps(X))
        stumps = []
        
        # vamos precisar selecionar alguns stumps
        if self.max_estimators < len(all_stumps):
            chosen_stump_indices = np.random.choice(len(all_stumps), size=self.max_estimators, replace=False)
            stumps = all_stumps[chosen_stump_indices].tolist()
        # quantidade de stumps solicitada é maior ou igual ao máximo
        else:
            stumps = all_stumps
            
        self.n_estimators = len(stumps)
        
        return stumps
    
    
    def _pick_best_stump(self, X, y):
        
        best_stump_id = -1
        best_stump_error = float("inf")
        best_stump_mistaken_sample_ids = []

        # percorre stumps e pega o que menos errou
        for stump_id, (feature, value, label) in enumerate(self.stumps):
            stump_error = 0
            mistaken_sample_ids = []

            # percorre os exemplos verificando se este stump errou cada um
            for sample_id, x in enumerate(X):
                # stump e exemplo têm mesma feature?
                feature_matches = x[feature] == value
                # stump e exemplo têm mesmo rótulo?
                label_matches = y[sample_id] == label

                if (feature_matches and not label_matches) or (not feature_matches and label_matches):
                    stump_error += self.input_weights[sample_id]
                    mistaken_sample_ids.append(True)
                else:
                    mistaken_sample_ids.append(False)


            if stump_error < best_stump_error:
                best_stump_error = stump_error
                best_stump_id = stump_id
                best_stump_mistaken_sample_ids = mistaken_sample_ids
        

        self.iteration_stumps.append(best_stump_id)
        self.iteration_errors.append(best_stump_error)
        
        return best_stump_error, best_stump_mistaken_sample_ids
    
    
    def _update_weights(self, error, mistaken_sample_ids):
        
        alpha = np.log((1-error)/error) / 2
        self.iteration_alphas.append(alpha)
        
        # acha novos pesos (sem normalizar ainda)        
        multiplier = (np.array(mistaken_sample_ids) * 2 - 1) * -1
        self.input_weights *= np.exp(-alpha * multiplier)

        # normaliza (divide por z)
        weight_sum = np.sum(self.input_weights, axis=0)
        self.input_weights = np.divide(self.input_weights, weight_sum)

    def get_params(self, deep=True):
        params = {
            "max_estimators": self.max_estimators
        }
        return params

    def __str__(self):
        return f"CoutoBoostClassifier(max_estimators={self.max_estimators})"

    def __repr__(self):
        return f"CoutoBoostClassifier(max_estimators={self.max_estimators})"
