import numpy as np


class CoutoBoostClassifier:
    """
    Implementação de AdaBoost baseada na
    interface pública dos classificadores
    do sk-learn.
    """
    def __init__(self, max_estimators=27):
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
        
        # Em alto nível:
        # 1. cria stumps (seguindo uma estratégia específica para tic-tac-toe)
        stumps = _create_stumps(X)
        
        # 2. inicializa escolhas das iterações
        self.iteration_errors = []
        self.iteration_alphas = []
        self.iteration_stumps = []
        
        # 3. cria pesos dos exemplos
        self.input_weights = np.full((1,X.shape[0]), 1/X.shape[0])
        
        # 4. repete
        for i in range(self.n_estimators):
            # 4.1. seleciona melhor stump (e o remove da lista de disponíveis)
            best_stump, error, mistakes = _pick_best_stump(X, y)
            
            # 4.2. calcula importância do stump selecionado
            # 4.3. atualiza os pesos considerando esse stump
            _update_weights(X, error, mistakes)
            
            # 4.4. se erro = 0 ou selecionou max_estimators, sai
            if error == 0:
                print("Saiu mais cedo porque erro empírico zerou na iteração " + str(i))
                break

        
        return self
        
        
        
    def predict(self, X):
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
        # Em alto nível
        # 1. invoca todos os stumps
        # 2. multiplica o resultado de cada um pelo alfa
        # 3. soma tudo e retorna
        value = 0
        classified = np.ndarray(shape=(X.shape, 1))
        for i in range(X.shape[0]):
            
            classified[i] = 0
            for t, (feature, value, label) in enumerate(self.iteration_stumps):
                alpha = self.iteration_alphas[t]
                
                if X[i, feature] == value:
                    classified[i] += alpha * label
                else:
                    classified[i] -= alpha * label
                
            if classified[i] > 0:
                classified[i] = 1
            elif classified[i] < 0:
                classified[i] = -1

            
        return classified
           
        
        
        
        
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
        all_stumps = _create_all_posible_stumps(X)
        stumps = []
        
        # vamos precisar selecionar alguns stumps
        if self.max_estimators < len(all_stumps):
            stumps = np.random.choice(all_stumps, self.max_estimators, replace=False)
        # pediu mais stumps do que tem como criar (sem redundância)
        else if self.max_estimators > len(all_stumps):
            print("Foi solicitada uma quantidade de estimadores maior do que possível")
            max_estimators = len(all_stumps)
            stumps = all_stumps
        # quantidade de stumps solicitada é igual ao máximo
        else:
            stumps = all_stumps
            
        self.n_estimators = len(stumps)
        
        return stumps
    
    
    def _pick_best_stump(self, X, y):
        
        best_stump_id = -1
        best_stump_error = float("inf")
        mistaken_sample_ids = []
        
        # percorre stumps e pega o que menos errou
        for stump_id, (feature, value, label) in enumerate(self.stumps):
            stump_error = 0
            
            # não colocar o mesmo stump mais de uma vez
            if stump_id in self.iteration_stumps:
                continue
            
            # percorre os exemplos verificando se este stump errou cada um
            for sample_id, x fom enumerate(X):
                # stump e exemplo têm mesma feature?
                feature_matches = x[feature] == value
                # stump e exemplo têm mesmo rótulo?
                label_matches = y[sample_id] == label
                
                if (feature_matches and not label_matches) or (not feature_matches and label_matches):
                    stump_error += self.input_weights[sample_id]
                    mistaken_sample_ids.append(sample_id)
            
            if stump_error < best_stump_error:
                best_stump_error = stump_error
                best_stump_id = stump_id
        

        self.iteration_stumps.append(best_stump)
        self.iteration_errors.append(error)
        
        return self.stumps[best_stump_id], best_stump_error, mistaken_sample_ids
    
    
    def _update_weights(self, X, error, mistaken_sample_ids):
        
        alpha = np.log((1-error)/error) / 2
        self.iteration_alphas.append(alpha)
        
        # acha novos pesos (sem normalizar ainda)
        weight_sum = 0
        for sample_i, weight in enumerate(self.input_weights):

            multiplier = 1
            if sample_i in mistaken_sample_ids:
                multiplier = -1
                
            new_weight = weight * np.exp(-alpha * multiplier)
            weight_sum += new_weight
            
            
        # normaliza (divide por z)
        self.input_weights = map(self.input_weights, lambda: w => w/weight_sum)

        