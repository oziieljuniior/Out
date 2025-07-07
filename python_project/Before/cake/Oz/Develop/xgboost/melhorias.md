Claro! Para melhorar o modelo XGBoost que você criou, podemos realizar várias análises e ajustes. Abaixo, vou detalhar algumas estratégias que você pode seguir para visualizar e melhorar o desempenho do modelo. Vou dividir isso em etapas claras e práticas.

---

### **1. Análise Inicial do Modelo**
Antes de fazer melhorias, é importante entender o desempenho atual do modelo. Você já está calculando a acurácia e o F1-Score, o que é ótimo. Vamos adicionar mais métricas e visualizações para ter uma visão mais completa.

#### a) **Matriz de Confusão**
A matriz de confusão ajuda a entender onde o modelo está errando (falsos positivos e falsos negativos).

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

#### b) **Relatório de Classificação**
O relatório de classificação fornece métricas como precisão, recall e F1-Score para cada classe.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

#### c) **Curva ROC e AUC**
A curva ROC e a área sob a curva (AUC) são úteis para avaliar a capacidade do modelo de distinguir entre as classes.

```python
from sklearn.metrics import roc_curve, auc

# Calcular as probabilidades das classes
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calcular a curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

---

### **2. Ajuste de Hiperparâmetros**
O ajuste de hiperparâmetros pode melhorar significativamente o desempenho do modelo. Vamos usar técnicas como **Grid Search** ou **Randomized Search** para encontrar os melhores hiperparâmetros.

#### a) **Grid Search**
O Grid Search testa todas as combinações possíveis de hiperparâmetros.

```python
from sklearn.model_selection import GridSearchCV

# Definir os parâmetros para o Grid Search
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200, 500],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 5, 10]
}

# Criar o modelo
model = xgb.XGBClassifier(objective='multi:softmax', num_class=2, random_state=42)

# Aplicar o Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)
grid_search.fit(X_train, y_train)

# Melhores parâmetros encontrados
print(f'Melhores parâmetros: {grid_search.best_params_}')

# Treinar o modelo com os melhores parâmetros
best_model = grid_search.best_estimator_
```

#### b) **Randomized Search**
O Randomized Search testa combinações aleatórias de hiperparâmetros, o que é mais eficiente em termos computacionais.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Definir os parâmetros para o Randomized Search
param_dist = {
    'learning_rate': uniform(0.001, 0.1),
    'max_depth': randint(3, 10),
    'n_estimators': randint(100, 1000),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'min_child_weight': randint(1, 10)
}

# Criar o modelo
model = xgb.XGBClassifier(objective='multi:softmax', num_class=2, random_state=42)

# Aplicar o Randomized Search
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, scoring='accuracy', cv=3, verbose=1)
random_search.fit(X_train, y_train)

# Melhores parâmetros encontrados
print(f'Melhores parâmetros: {random_search.best_params_}')

# Treinar o modelo com os melhores parâmetros
best_model = random_search.best_estimator_
```

---

### **3. Validação Cruzada**
A validação cruzada ajuda a garantir que o modelo generalize bem para dados não vistos.

```python
from sklearn.model_selection import cross_val_score

# Aplicar validação cruzada
scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')

# Média e desvio padrão das pontuações
print(f'Acurácia média: {scores.mean():.4f}')
print(f'Desvio padrão: {scores.std():.4f}')
```

---

### **4. Feature Engineering**
Melhorar as características (features) pode aumentar o desempenho do modelo. Algumas técnicas incluem:
- **Seleção de características:** Usar métodos como `SelectKBest` ou `RFE` (Recursive Feature Elimination).
- **Criação de novas características:** Combinar ou transformar características existentes para capturar padrões adicionais.

---

### **5. Visualização de Melhorias**
Após ajustar o modelo, você pode comparar as métricas antes e depois das melhorias. Por exemplo:
- Compare a acurácia, F1-Score, AUC e matriz de confusão antes e depois do ajuste de hiperparâmetros.
- Use gráficos para visualizar as diferenças.

---

### **6. Exemplo de Código Completo**
Aqui está um exemplo de como você pode integrar todas essas etapas:

```python
# Treinar o modelo inicial
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=2,
    eval_metric='mlogloss',
    learning_rate=0.005,
    n_estimators=500,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.2,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
model.fit(X_train, y_train)

# Avaliar o modelo inicial
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Aplicar Grid Search ou Randomized Search
# (Use o código de Grid Search ou Randomized Search fornecido acima)

# Avaliar o modelo ajustado
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best))

# Visualizar a importância das características
xgb.plot_importance(best_model)
plt.show()
```

---

Com essas análises e ajustes, você poderá visualizar e melhorar o desempenho do seu modelo XGBoost. Se precisar de mais ajuda ou tiver dúvidas específicas, é só perguntar! 😊