from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Any
from .get_info import get_info
from .dynamic_change import dynamic_change
from .utils import pdist2, resolve_data_file, load_mat_variable
from ..boundary_check import boundary_check
from scipy.io import loadmat

# Sub-functions
from .sub_f.ef8f2 import ef8f2
from .sub_f.fgrienwank import fgrienwank
from .sub_f.fweierstrass import fweierstrass
from .sub_f.frastrigin import frastrigin
from .sub_f.fsphere import fsphere


@dataclass
class DMMOP:
    fun_num: int = 1
    change_type: int = 1
    D: int = 5
    lower: np.ndarray = field(default_factory=lambda: None)
    upper: np.ndarray = field(default_factory=lambda: None)
    freq: int = 0
    evaluation: int = 0
    evaluated: int = 0
    rs: Dict[str, np.ndarray] = field(default_factory=dict)
    env: int = 0
    maxEnv: int = 60
    change: bool = False

    # DF1 parameters
    ln: int = 0
    h: np.ndarray | None = None
    w: np.ndarray | None = None
    h_s: float = 7.0
    w_s: float = 1.0
    h_bound: Tuple[float, float] = (30.0, 70.0)
    w_bound: Tuple[float, float] = (1.0, 12.0)

    # Composition parameters
    M: List[np.ndarray] | None = None
    M_initial: List[np.ndarray] | None = None
    M_angel: np.ndarray | None = None
    sub_fun: List[Callable[[np.ndarray], np.ndarray]] | None = None
    sigma: np.ndarray | None = None
    lambda_: np.ndarray | None = None

    # Peaks and history
    gn_max: int = 4
    dpeaks: float = 0.1
    selected_idx: np.ndarray | None = None
    is_add: bool = False
    gn: int = 0
    x: np.ndarray | None = None
    x_initial: np.ndarray | None = None
    x_angel: float = 0.0
    o: np.ndarray | None = None
    of: np.ndarray | None = None
    his_o: List[Tuple[int, int, np.ndarray]] = field(default_factory=list)
    his_of: List[Tuple[int, int, np.ndarray]] = field(default_factory=list)
    data: List[Dict[str, Any]] = field(default_factory=list)

    # RNG
    proRand: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def __init__(self, fn: int):
        fun_num, change_type, D = get_info(fn)
        self.fun_num = fun_num
        self.change_type = change_type
        self.D = D
        self.lower = -5.0 * np.ones(D)
        self.upper = 5.0 * np.ones(D)
        self.freq = 5000 * D
        self.maxEnv = 60
        self.evaluation = self.maxEnv * self.freq
        self.evaluated = 0
        self.rs = {}
        self.env = 0
        self.change = False
        self.proRand = np.random.default_rng(0)
        self.his_o = []
        self.his_of = []
        self.data = [None] * self.maxEnv
        self.Initialize(0)

    def Initialize(self, evaluated: int):
        if self.fun_num <= 4:
            h_global = 75.0
            if self.fun_num == 1:
                self.ln = 4
                self.h = np.array([*([h_global] * self.gn_max), 62.5889, 66.2317, 68.0795, 66.5350], dtype=float)
                self.w = np.array([7.9560, 2.0729, 4.0635, 7.0157, 11.5326, 11.6138, 2.7337, 11.6765], dtype=float)
                t = loadmat(resolve_data_file('F1X.mat'))
                data = t.get('data')
                if data is None:
                    raise KeyError('F1X.mat missing variable "data"')
                self.x = data[: self.gn_max + self.ln, : self.D]
            elif self.fun_num == 2:
                if self.gn_max % 2 == 1:
                    self.gn_max += 1
                self.ln = 0
                self.h = np.ones(self.gn_max) * h_global
                self.w = np.ones(self.gn_max + self.ln) * 12.0
                self.x = np.tile(np.array([-3, -2, 2, 3])[:, None], (1, self.D))
            elif self.fun_num == 3:
                self.ln = 0
                self.h = np.ones(self.gn_max) * h_global
                self.w = np.ones(self.gn_max + self.ln) * 5.0
                self.x = np.tile((np.array([-3, -2, 0, 4]) + 0.5)[:, None], (1, self.D))
            elif self.fun_num == 4:
                self.ln = 0
                self.h = np.ones(self.gn_max) * h_global
                self.w = np.ones(self.gn_max) * 5.0
                self.x = np.tile(np.array([-3, -1, 1, 3])[:, None], (1, self.D))
            else:
                raise ValueError('fun_num in DMMOP is illegal.')
            self.x = self.SetMinDistance(self.x)
            self.x_initial = self.x.copy()
            if self.change_type in (5, 6):
                # heights (locals) and widths change
                self.h = np.concatenate((np.ones(self.gn_max) * h_global, dynamic_change(self.proRand, self.h[self.gn_max:], self.change_type, self.h_bound[0], self.h_bound[1], self.h_s, self.env)))
                self.w = dynamic_change(self.proRand, self.w, self.change_type, self.w_bound[0], self.w_bound[1], self.w_s, self.env)
                self.x, self.x_angel = self.ChangeMatrix(self.x, self.x_initial, self.lower, self.upper, self.x_angel, 'x')
            self.x = self.SetMinDistance(self.x)
        else:
            self.ln = 0
            if self.fun_num == 5:
                self.sub_fun = [fgrienwank, fgrienwank, fweierstrass, fweierstrass, fsphere, fsphere]
                self.gn_max = len(self.sub_fun)
                self.sigma = np.ones(self.gn_max)
                self.lambda_ = np.array([1, 1, 8, 8, 1 / 5, 1 / 5], dtype=float)
                self.M = [np.eye(self.D) for _ in range(self.gn_max)]
            elif self.fun_num == 6:
                self.sub_fun = [frastrigin, frastrigin, fweierstrass, fweierstrass, fgrienwank, fgrienwank, fsphere, fsphere]
                self.gn_max = len(self.sub_fun)
                self.sigma = np.ones(self.gn_max)
                self.lambda_ = np.array([1, 1, 10, 10, 1 / 10, 1 / 10, 1 / 7, 1 / 7], dtype=float)
                self.M = [np.eye(self.D) for _ in range(self.gn_max)]
            elif self.fun_num == 7:
                self.sub_fun = [ef8f2, ef8f2, fweierstrass, fweierstrass, fgrienwank, fgrienwank]
                self.gn_max = len(self.sub_fun)
                self.sigma = np.array([1, 1, 2, 2, 2, 2], dtype=float)
                self.lambda_ = np.array([1 / 4, 1 / 10, 2, 1, 2, 5], dtype=float)
                if self.D not in (2, 3, 5, 10, 20):
                    raise ValueError('the dimension is illegal in function 7.')
                t = loadmat(resolve_data_file(f'CF3_M_D{self.D}.mat'))
                if 'M' not in t:
                    self.M = None
                else:
                    Marr = t['M']
                    if Marr.ndim == 2 and Marr.shape[1] == 1:
                        self.M = [np.array(Marr[i, 0]) for i in range(Marr.shape[0])]
                    elif Marr.ndim == 2 and Marr.shape[0] == 1:
                        self.M = [np.array(Marr[0, i]) for i in range(Marr.shape[1])]
                    else:
                        self.M = [np.array(Marr.flat[i]) for i in range(Marr.size)]
            elif self.fun_num == 8:
                self.sub_fun = [frastrigin, frastrigin, ef8f2, ef8f2, fweierstrass, fweierstrass, fgrienwank, fgrienwank]
                self.gn_max = len(self.sub_fun)
                self.sigma = np.array([1, 1, 1, 1, 1, 2, 2, 2], dtype=float)
                self.lambda_ = np.array([4, 1, 4, 1, 1 / 10, 1 / 5, 1 / 10, 1 / 40], dtype=float)
                if self.D not in (2, 3, 5, 10, 20):
                    raise ValueError('the dimension is illegal in function 8.')
                t = loadmat(resolve_data_file(f'CF4_M_D{self.D}.mat'))
                if 'M' not in t:
                    self.M = None
                else:
                    Marr = t['M']
                    if Marr.ndim == 2 and Marr.shape[1] == 1:
                        self.M = [np.array(Marr[i, 0]) for i in range(Marr.shape[0])]
                    elif Marr.ndim == 2 and Marr.shape[0] == 1:
                        self.M = [np.array(Marr[0, i]) for i in range(Marr.shape[1])]
                    else:
                        self.M = [np.array(Marr.flat[i]) for i in range(Marr.size)]
            else:
                raise ValueError('fun_num in DMMOP is illegal.')
            t = loadmat(resolve_data_file('optima.mat'))
            o = t.get('o')
            if o is None:
                raise KeyError('optima.mat missing variable "o"')
            self.x = o[::-1][: self.gn_max, : self.D]
            self.x = self.SetMinDistance(self.x)
            self.x_initial = self.x.copy()
            self.M_initial = [m.copy() for m in self.M]
            self.M_angel = np.zeros(len(self.M))
            if self.change_type in (5, 6):
                self.x, self.x_angel = self.ChangeMatrix(self.x, self.x_initial, self.lower, self.upper, self.x_angel, 'x')
                for i in range(len(self.M)):
                    self.M[i], self.M_angel[i] = self.ChangeMatrix(self.M[i], self.M_initial[i], -np.ones(self.D), np.ones(self.D), self.M_angel[i], f'M{i+1}')
                    if not np.allclose(np.round(self.M[i] @ self.M[i].T), np.eye(self.M[i].shape[0])):
                        raise ValueError('no orth matrix')
            self.x = self.SetMinDistance(self.x)

        if self.change_type in (7, 8):
            # MATLAB: randi(proRand, gn_max-1)+1 generates [1, gn_max-1] + 1 = [2, gn_max]
            # Python: integers(1, gn_max) generates [1, gn_max-1], then +1 gives [2, gn_max]
            self.gn = self.proRand.integers(1, self.gn_max) + 1  # 2..gn_max (matches MATLAB)
            sel = np.sort(self.proRand.choice(self.gn_max, size=self.gn, replace=False))
            self.selected_idx = np.concatenate([sel, np.arange(self.gn_max, self.gn_max + self.ln)])
            if self.change_type == 7 and self.gn == 2:
                self.is_add = True
        else:
            self.selected_idx = np.arange(self.gn_max + self.ln)
            self.gn = self.gn_max

        self.o = self.x[self.selected_idx[: self.gn], :]
        self.of = self.GetFitsCore(self.o)
        self.his_o.append((self.env, evaluated, self.o.copy()))
        self.his_of.append((self.env, evaluated, self.of.copy()))

    def GetFits(self, decs: np.ndarray) -> np.ndarray:
        if self.change:
            return np.array([])
        rest = self.freq - (self.evaluated % self.freq)
        num = int(min(rest, decs.shape[0]))
        fits = self.GetFitsCore(decs[:num, :])
        self.evaluated += num
        self.change = (num == rest)
        return fits

    def GetFitsCore(self, decs: np.ndarray) -> np.ndarray:
        if self.fun_num <= 4:
            dist = pdist2(decs, self.x[self.selected_idx, :])
            h = self.h[self.selected_idx]
            w = self.w[self.selected_idx]
            return np.max(h[None, :] - w[None, :] * dist, axis=1)
        else:
            return self.hybrid_composition_func(
                decs,
                self.gn,
                [self.sub_fun[i] for i in self.selected_idx[: self.gn]],
                self.x[self.selected_idx[: self.gn], :],
                self.sigma[self.selected_idx[: self.gn]],
                self.lambda_[self.selected_idx[: self.gn]],
                np.zeros(self.gn),
                [self.M[i] for i in self.selected_idx[: self.gn]],
            )

    def Terminate(self) -> bool:
        return self.evaluated >= self.evaluation

    def CheckChange(self, pop: np.ndarray, fits: np.ndarray) -> bool:
        chenv = self.change
        if self.change:
            # store environment data at index env (0-based -> env)
            idx = self.env
            self.data[idx] = {
                'pop': pop.copy(),
                'fits': fits.copy(),
                'o': self.o.copy(),
                'of': self.of.copy(),
                'eva': self.evaluated,
            }
            self.ChangeDynamic()
            self.change = False
        return chenv

    def GetPeak(self) -> Tuple[np.ndarray, np.ndarray]:
        catchpeak = np.zeros((3, self.maxEnv), dtype=int)
        allpeak = np.zeros(self.maxEnv, dtype=int)
        e_d = 0.1 / 2.0
        for i in range(self.maxEnv):
            entry = self.data[i]
            if entry is None:
                continue
            result_pop = entry['pop']
            result_fit = entry['fits']
            result_of = entry['of'][0]
            result_o = entry['o']
            allpeak[i] = result_o.shape[0]
            for j in range(3):
                select = (result_of - result_fit) < 10 ** (-(j + 2))
                select_pop = result_pop[select, :]
                if select_pop.size == 0:
                    catchpeak[j, i] = 0
                else:
                    best_dis = pdist2(result_o, select_pop)
                    found = np.min(best_dis, axis=1) < e_d
                    catchpeak[j, i] = int(np.sum(found))
        return catchpeak, allpeak

    def GetBestFit(self):
        return self.his_of

    def GetBestPos(self):
        return self.his_o

    def ChangeDynamic(self):
        if self.evaluated < self.evaluation:
            self.env += 1
            if self.change_type == 7:
                if self.is_add:
                    self.gn += 1
                    if self.gn == self.gn_max:
                        self.is_add = False
                else:
                    self.gn -= 1
                    if self.gn == 2:
                        self.is_add = True
                self.selected_idx = np.arange(self.gn)
            elif self.change_type == 8:
                self.gn = int(self.proRand.integers(1, self.gn_max))
                self.selected_idx = np.sort(self.proRand.choice(self.gn_max, size=self.gn, replace=False))
            if self.change_type in (7, 8):
                self.selected_idx = np.concatenate([self.selected_idx, np.arange(self.gn_max, self.gn_max + self.ln)])
                # adopt C1 to change parameters
                t = self.change_type
                self.change_type = 1
                self.env -= 1
                self.ChangeDynamic()
                self.change_type = t
                return
            if self.fun_num <= 4:
                self.h[self.gn_max:] = dynamic_change(self.proRand, self.h[self.gn_max:], self.change_type, self.h_bound[0], self.h_bound[1], self.h_s, self.env)
                self.w = dynamic_change(self.proRand, self.w, self.change_type, self.w_bound[0], self.w_bound[1], self.w_s, self.env)
            else:
                for i in range(len(self.M)):
                    self.M[i], self.M_angel[i] = self.ChangeMatrix(self.M[i], self.M_initial[i], -np.ones(self.D), np.ones(self.D), self.M_angel[i], f'M{i+1}')
                    if not np.allclose(np.round(self.M[i] @ self.M[i].T), np.eye(self.M[i].shape[0])):
                        raise ValueError('no orth matrix')
            self.x, self.x_angel = self.ChangeMatrix(self.x, self.x_initial, self.lower, self.upper, self.x_angel, 'x')
            self.x = self.SetMinDistance(self.x)
            self.o = self.x[self.selected_idx[: self.gn], :]
            self.of = self.GetFitsCore(self.o)
            self.his_o.append((self.env, self.evaluated, self.o.copy()))
            self.his_of.append((self.env, self.evaluated, self.of.copy()))

    # Private helpers
    def hybrid_composition_func(self, x: np.ndarray, func_num: int, func: List[Callable[[np.ndarray], np.ndarray]], o: np.ndarray, sigma: np.ndarray, lambda_: np.ndarray, bias: np.ndarray, M: List[np.ndarray]) -> np.ndarray:
        ps, d = x.shape
        weight = np.zeros((ps, func_num))
        for i in range(func_num):
            oo = np.tile(o[i, :], (ps, 1))
            weight[:, i] = np.exp(-np.sum((x - oo) ** 2, axis=1) / 2.0 / (d * sigma[i] ** 2))
        max_weight = np.max(weight, axis=1)
        weight = (weight == max_weight[:, None]) * weight + (weight != max_weight[:, None]) * (weight * (1 - (max_weight[:, None] ** 10)))
        zero_rows = np.sum(weight, axis=1) == 0
        if np.any(zero_rows):
            weight[zero_rows, :] += 1.0
        weight = weight / np.sum(weight, axis=1, keepdims=True)
        res = np.zeros(ps)
        for i in range(func_num):
            oo = np.tile(o[i, :], (ps, 1))
            Mi = M[i]
            xi = ((x - oo) / lambda_[i]) @ Mi
            if xi.ndim == 1:
                xi = xi[None, :]
            f = func[i](xi)
            x1 = self.upper
            x1_row = (x1 / lambda_[i]) @ Mi
            f1 = func[i](x1_row[None, :])
            fit1 = 2000.0 * f / f1
            res += weight[:, i] * (fit1 + bias[i])
        return res * (-1.0)

    def ChangeMatrix(self, m: np.ndarray, m_initial: np.ndarray, lb: np.ndarray, ub: np.ndarray, old_theta: float, name: str) -> Tuple[np.ndarray, float]:
        d = m.shape[1]
        l = d - 1 if d % 2 == 1 else d
        r = self.proRand.choice(d, size=l, replace=False)
        if self.change_type in (5, 6):
            phi_min = 0.0
            phi_max = np.pi / 6.0
            m = m_initial.copy()
            if self.env >= 12 and name in self.rs:
                r = self.rs[name][self.env % 12, :]
        else:
            phi_min = -np.pi
            phi_max = np.pi
        if name not in self.rs:
            self.rs[name] = np.zeros((self.maxEnv, l), dtype=int)
        self.rs[name][self.env, :] = r
        if self.change_type in (1, 2, 3):
            old_theta = 0.0
        theta = float(dynamic_change(self.proRand, np.array([old_theta]), self.change_type, phi_min, phi_max, 1.0, self.env)[0])
        rotation_M = np.eye(d)
        for i in range(l // 2):
            a, b = int(r[i * 2]), int(r[i * 2 + 1])
            rotation_M[a, a] = np.cos(theta)
            rotation_M[b, b] = np.cos(theta)
            rotation_M[a, b] = -np.sin(theta)
            rotation_M[b, a] = np.sin(theta)
        new_m = m @ rotation_M
        new_m = boundary_check(new_m, lb, ub)
        return new_m, theta

    def SetMinDistance(self, pos: np.ndarray) -> np.ndarray:
        pos = pos.copy()
        for i in range(pos.shape[0]):
            accept = False
            if i == 0:
                accept = True
            else:
                dis = pdist2(pos[i:i+1, :], pos[:i, :]).ravel()
                if np.all(dis > self.dpeaks):
                    accept = True
            while not accept:
                v = self.proRand.random(self.D) - 0.5
                length = np.sqrt(np.sum(v ** 2))
                v = v * (self.dpeaks / length)
                pos[i, :] = pos[i, :] + v
                pos[i, :] = boundary_check(pos[i:i+1, :], self.lower, self.upper)[0]
                dis = pdist2(pos[i:i+1, :], pos[:i, :]).ravel()
                if np.all(dis > self.dpeaks):
                    accept = True
        return pos
