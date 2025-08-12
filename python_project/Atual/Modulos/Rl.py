"""
BanditAgent: agente de Bandit Contextual (Thompson Sampling Logístico via ensemble bootstrap)
com recompensa atrasada (1 passo) e "gating" com o seu modelo base.

Uso típico no seu loop online:

    from BanditAgent import BanditAgent, BanditConfig

    agent = BanditAgent(cfg=BanditConfig(C_FP=4.0, G_TP=1.0, gamma=0.999, target_selection=0.33),
                        limiar_odd=3.0, seed=123)

    # a cada iteração t (após ter o vetor X_base e a próx. odd real disponível):
    action, info = agent.step(
        x_base=X_base_vector_1d,      # np.ndarray 1D do seu Vetores.transformar_entrada_predicao(...).ravel()
        proba_pred=float(p_model),    # prob. do seu classificador (para odds > limiar)
        model_gt1=int(model_sugere_gt),  # 1 se o modelo base sugere "agir" (ex.: odds > limiar), senão 0
        odd_current=float(odd_atual)  # odd atual observada (usada p/ atualizar a decisão ANTERIOR)
        # extras=np.array([...], dtype=np.float32)  # opcional: features extras do seu histórico
    )

    # 'action' é a decisão FINAL (gating: modelo base E bandit concordam).
    # 'info' traz métricas on-line e p_TS, threshold, etc.

Detalhes:
- Recompensa atrasada (1 passo): a decisão do passo t-1 é atualizada
  quando você chama step(...) em t, passando a odd_current (verdade observável).
- y_prev = 1 se odd_current > limiar_odd, senão 0 (classe operacional "odds > limiar").
- O contexto x_t = concat([x_base, proba_pred, model_gt1, extras]).
- O agente inicializa automaticamente na primeira chamada (d detectado).
- Apenas NumPy como dependência externa.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

# ------------------------- Utils -------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))

# ------------------------- Métricas -------------------------

@dataclass
class Metrics:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    acted: int = 0
    total: int = 0

    def update(self, action: int, y: int):
        self.total += 1
        if action == 1:
            self.acted += 1
            if y == 1:
                self.tp += 1
            else:
                self.fp += 1
        else:
            if y == 0:
                self.tn += 1
            else:
                self.fn += 1

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def fpr(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    @property
    def selection_rate(self) -> float:
        return self.acted / self.total if self.total > 0 else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "fpr": self.fpr,
            "selection_rate": self.selection_rate,
            "tp": float(self.tp),
            "tn": float(self.tn),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "total": float(self.total),
        }

# ------------------------- Online Logistic (base) -------------------------

class _OnlineLogistic:
    """Regressão logística on-line com AdaGrad + L2 + esquecimento (gamma)."""
    def __init__(self, d: int, lr: float=0.05, l2: float=1e-4, gamma: float=0.999, seed: Optional[int]=None):
        self.d = d
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.01, size=d)
        self.lr = lr
        self.l2 = l2
        self.gamma = gamma
        self._G = np.full(d, 1e-8)  # acumulador AdaGrad

    def predict_proba(self, x: np.ndarray) -> float:
        return float(_sigmoid(np.dot(self.w, x)))

    def update(self, x: np.ndarray, y: int, k: int=1):
        if k <= 0:
            return
        for _ in range(k):
            self.w *= self.gamma
            p = self.predict_proba(x)
            grad = (p - y) * x + self.l2 * self.w
            self._G += grad * grad
            adj_lr = self.lr / (np.sqrt(self._G) + 1e-8)
            self.w -= adj_lr * grad

# ------------------------- Ensemble TS (bootstrap) -------------------------

class BootstrappedLogisticTS:
    """Thompson Sampling aproximado por ensemble bootstrap.
    - K cabeças de regressão logística on-line
    - Em cada rodada, amostra UMA cabeça para decidir (TS)
    - Atualização com multiplicidade Poisson(1) por cabeça (bootstrap on-line)
    """
    def __init__(self, d: int, K: int=10, lr: float=0.05, l2: float=1e-4, gamma: float=0.999, seed: int=42):
        self.d = d
        self.K = K
        self.rng = np.random.default_rng(seed)
        self.heads = [
            _OnlineLogistic(d, lr=lr, l2=l2, gamma=gamma, seed=int(seed + i))
            for i in range(K)
        ]
        self._last_head_idx = 0

    def sample_head(self) -> int:
        h = int(self.rng.integers(0, self.K))
        self._last_head_idx = h
        return h

    def sample_p_success(self, x: np.ndarray) -> float:
        h = self.sample_head()
        return self.heads[h].predict_proba(x)

    def decide(self, x: np.ndarray, threshold: float, margin: float=0.0) -> Tuple[int, float]:
        p = self.sample_p_success(x)
        action = 1 if (p > (threshold + margin)) else 0
        return action, p

    def update(self, x: np.ndarray, y: int, action: int):
        # full-information: sabemos y de qualquer forma no seu cenário
        for head in self.heads:
            k = int(self.rng.poisson(1.0))
            head.update(x, y, k=k)

# ------------------------- Calibrador de taxa de ação -------------------------

class ThresholdCalibrator:
    def __init__(self, target_rate: float, step: float=1e-3, tol: float=0.02, min_thr: float=0.01, max_thr: float=0.99):
        self.target = target_rate
        self.step = step
        self.tol = tol
        self.min_thr = min_thr
        self.max_thr = max_thr
        self.threshold = 0.5
        self._win_actions = []  # janela móvel de ações

    def update_and_get(self, acted: int, win: int=600) -> float:
        self._win_actions.append(int(acted))
        if len(self._win_actions) > win:
            self._win_actions.pop(0)
        rate = np.mean(self._win_actions) if self._win_actions else 0.0
        if rate < self.target - self.tol:
            self.threshold = min(self.max_thr, self.threshold - self.step)
        elif rate > self.target + self.tol:
            self.threshold = max(self.min_thr, self.threshold + self.step)
        return self.threshold

# ------------------------- Config e Agente -------------------------

@dataclass
class BanditConfig:
    K: int = 10
    lr: float = 0.05
    l2: float = 1e-4
    gamma: float = 0.999
    C_FP: float = 4.0     # custo de falso positivo (alto => mais conservador)
    G_TP: float = 1.0     # ganho por verdadeiro positivo
    margin: float = 0.0   # margem extra na decisão do bandit
    target_selection: Optional[float] = 0.33  # alvo ~33%; None para desativar

class BanditAgent:
    def __init__(self, cfg: BanditConfig=BanditConfig(), limiar_odd: float=3.0, seed: int=42):
        self.cfg = cfg
        self.limiar_odd = float(limiar_odd)
        self.rng = np.random.default_rng(seed)

        self.metrics = Metrics()
        self.calibrator = ThresholdCalibrator(cfg.target_selection) if cfg.target_selection is not None else None
        self.bandit: Optional[BootstrappedLogisticTS] = None
        self.base_threshold = self._cost_threshold(cfg.C_FP, cfg.G_TP)

        # estado de recompensa atrasada
        self._pending: Optional[Tuple[np.ndarray, int]] = None  # (x_prev, action_prev)

    @staticmethod
    def _cost_threshold(C_FP: float, G_TP: float) -> float:
        if C_FP <= 0 and G_TP <= 0:
            return 1.0
        return C_FP / (C_FP + max(G_TP, 1e-12))

    def _ensure_init(self, d_total: int, seed: int=42):
        if self.bandit is None:
            self.bandit = BootstrappedLogisticTS(
                d=d_total, K=self.cfg.K, lr=self.cfg.lr, l2=self.cfg.l2, gamma=self.cfg.gamma, seed=seed
            )

    def step(self,
             x_base: np.ndarray,
             proba_pred: float,
             model_gt1: int,
             odd_current: Optional[float]=None,
             extras: Optional[np.ndarray]=None,
             window_for_calib: int=600) -> Tuple[int, Dict[str, float]]:
        """Executa UM passo do agente.
        - x_base: vetor 1D (np.ndarray) das suas features do Vetores.transformar_entrada_predicao(...).ravel()
        - proba_pred: probabilidade do seu modelo base para a classe "agir" (odds > limiar)
        - model_gt1: 1 se seu modelo base sugere agir; 0 caso contrário
        - odd_current: odd atual observada (serve para atualizar a decisão PENDENTE do passo anterior)
        - extras: vetor 1D opcional com features adicionais
        - window_for_calib: tamanho da janela para o calibrador de taxa de ação
        Retorna: (final_action, info)
        """
        x_base = np.asarray(x_base, dtype=np.float32).ravel()
        if extras is not None:
            extras = np.asarray(extras, dtype=np.float32).ravel()
            x_t = np.concatenate([x_base, np.array([proba_pred, float(model_gt1)], dtype=np.float32), extras], axis=0)
        else:
            x_t = np.concatenate([x_base, np.array([proba_pred, float(model_gt1)], dtype=np.float32)], axis=0)

        # inicializa bandit quando soubermos d_total
        self._ensure_init(d_total=x_t.shape[0], seed=123)

        # 1) Atualiza a decisão ANTERIOR com a verdade atual (recompensa atrasada 1 passo)
        if self._pending is not None and odd_current is not None:
            y_prev = 1 if float(odd_current) > self.limiar_odd else 0
            x_prev, action_prev = self._pending
            # update bandit e métricas
            self.bandit.update(x_prev, y_prev, action_prev)  # type: ignore[arg-type]
            self.metrics.update(action_prev, y_prev)
            self._pending = None

        # 2) Decide no passo atual (Thompson amostrado)
        thr_util = self.base_threshold if self.calibrator is None else self.calibrator.threshold
        action_bandit, p_ts = self.bandit.decide(x_t, threshold=thr_util, margin=self.cfg.margin)  # type: ignore[union-attr]

        # Gating com o modelo base: só age se ambos concordarem
        final_action = 1 if (int(model_gt1) == 1 and action_bandit == 1) else 0

        # 3) Agenda atualização para a próxima rodada
        self._pending = (x_t, final_action)

        # 4) Atualiza calibrador de taxa de ação (se existir)
        if self.calibrator is not None:
            self.calibrator.update_and_get(final_action, win=window_for_calib)
            thr_util = self.calibrator.threshold

        # 5) Monta info
        info = {
            "p_TS": float(p_ts),
            "threshold_util": float(thr_util),
            "final_action": float(final_action),
        }
        info.update(self.metrics.as_dict())
        return final_action, info

    def flush_pending_with(self, odd_current: float) -> Optional[Dict[str, float]]:
        """Se quiser forçar o update da decisão pendente (por exemplo ao encerrar o loop)."""
        if self._pending is None:
            return None
        y_prev = 1 if float(odd_current) > self.limiar_odd else 0
        x_prev, action_prev = self._pending
        self.bandit.update(x_prev, y_prev, action_prev)  # type: ignore[union-attr]
        self.metrics.update(action_prev, y_prev)
        self._pending = None
        return self.metrics.as_dict()

    def get_metrics(self) -> Dict[str, float]:
        return self.metrics.as_dict()

    def current_threshold(self) -> float:
        return self.base_threshold if self.calibrator is None else self.calibrator.threshold

"""
Notas de integração rápida com o seu Producao.py:

1) Após montar `Apredicao = vetores.transformar_entrada_predicao(arrayodd)` (shape (1, d_base)) e `proba_pred = logreg.predict_proba(Apredicao)[:, 1]`:

   - Converta para 1D: `x_base = Apredicao.ravel().astype(np.float32)`
   - Defina a classe operacional do seu modelo base (ex.: odds > limiar):
       `model_gt1 = int(1 - yhat_lr[0])`  # se o seu y de treino é 1 para "< limiar"

2) Crie o agente UMA vez (fora do loop) e chame `agent.step(...)` a cada iteração,
   passando `odd_current = odd` (a odd observada naquela iteração) para atualizar a decisão anterior.

3) Use `final_action` como sua decisão FINAL (0/1) para alimentar o seu Placar/score.
   As métricas on-line ficam em `info` e também em `agent.get_metrics()`.

4) Ao encerrar o loop, se sobrar `_pending`, chame `agent.flush_pending_with(odd_final)`
   para atualizar a última decisão.
"""
