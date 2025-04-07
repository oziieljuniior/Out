import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    precision_score,
    recall_score
)

class RedeNeuralXGBoost:
    """
    Classe aprimorada para treinamento e avaliação de um modelo XGBoost com foco no balanceamento de classes.
    
    Melhorias incluem:
    - Controle de threshold de classificação
    - Balanceamento automático de classes
    - Métricas adicionais de avaliação
    - Amostragem estratificada
    """

    def __init__(self, learning_rate=0.01, n_estimators=50, max_depth=25, class_ratio=None):
        """
        Inicializa o modelo com hiperparâmetros personalizáveis e controle de balanceamento.
        
        Args:
            learning_rate (float): Taxa de aprendizado (padrão: 0.005)
            n_estimators (int): Número de árvores (padrão: 500)
            max_depth (int): Profundidade máxima das árvores (padrão: 25)
            class_ratio (tuple): Proporção esperada das classes (classe 1, classe 0). Ex: (0.3, 0.7)
        """
        self.class_ratio = class_ratio
        scale_pos_weight = class_ratio[1]/class_ratio[0] if class_ratio else 1
        
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False
        )
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.best_threshold = 0.75

    def treinar(self, X, y, test_size=0.2, random_state=42):
        """
        Treina o modelo com amostragem estratificada para manter proporção das classes.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Distribuição original - 1: {np.mean(y):.4f}, 0: {1-np.mean(y):.4f}")
        print(f"Distribuição treino - 1: {np.mean(self.y_train):.4f}, 0: {1-np.mean(self.y_train):.4f}")
        
        self.model.fit(self.X_train, self.y_train)
        
        # Encontra o melhor threshold se class_ratio foi especificado
        if self.class_ratio:
            self.ajustar_threshold_otimo()
            
        return self.model

    def ajustar_threshold_otimo(self):
        """Encontra o threshold que melhor aproxima a distribuição desejada."""
        y_proba = self.model.predict_proba(self.X_train)[:, 1]
        
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.5
        best_diff = float('inf')
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            ratio = np.mean(y_pred)
            diff = abs(ratio - self.class_ratio[0])
            
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
                
        self.best_threshold = best_threshold
        print(f"Threshold ótimo ajustado para: {best_threshold:.4f}")

    def avaliar(self, threshold=None):
        """Avaliação completa com métricas balanceadas."""
        if threshold is None:
            threshold = self.best_threshold
        
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # Métricas básicas
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # Distribuição das predições
        pred_ratio = np.mean(y_pred)
        
        print("\n=== Métricas de Avaliação ===")
        print(f"Threshold usado: {threshold:.4f}")
        print(f"Distribuição Predita - 1: {pred_ratio:.4f}, 0: {1-pred_ratio:.4f}")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"Precision (1): {precision:.4f}")
        print(f"Recall (1): {recall:.4f}")
        print(f"F1-Score (1): {f1:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(self.y_test, y_pred))
        
        # Gráficos
        self._plot_importance()
        self._plot_confusion_matrix(y_pred)
        self._plot_roc_curve(y_proba)
        
        return {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pred_ratio': pred_ratio
        }

    def _plot_importance(self):
        """Plota a importância das features."""
        xgb.plot_importance(self.model)
        plt.title("Importância das Features")
        plt.show()

    def _plot_confusion_matrix(self, y_pred):
        """Plota a matriz de confusão."""
        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Matriz de Confusão")
        plt.show()

    def _plot_roc_curve(self, y_proba):
        """Plota a curva ROC."""
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
        plt.xlabel('Falso Positivo')
        plt.ylabel('Verdadeiro Positivo')
        plt.title('Curva ROC')
        plt.legend()
        plt.show()

    def avaliar_limiar(self, thresholds=np.linspace(0.1, 0.9, 9)):
        """Avalia o modelo em diferentes thresholds."""
        results = []
        for threshold in thresholds:
            print(f"\n=== Avaliação com threshold {threshold:.2f} ===")
            metrics = self.avaliar(threshold=threshold)
            results.append(metrics)
        return results


# Exemplo de uso:
if __name__ == "__main__":
    # Dados de exemplo (substitua pelos seus arrays)
    np.random.seed(42)
    X = np.random.rand(1000, 10)  # 1000 amostras, 10 features
    y = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])  # Distribuição 30%-70%

    # Uso da classe com controle de balanceamento
    print("\n=== Modelo com Balanceamento ===")
    rede = RedeNeuralXGBoost(learning_rate=0.01, 
                           n_estimators=300,
                           class_ratio=(0.3, 0.7))
    rede.treinar(X, y)
    
    # Avaliação padrão com threshold ótimo
    rede.avaliar()
    
    # Avaliação em múltiplos thresholds
    rede.avaliar_limiar()